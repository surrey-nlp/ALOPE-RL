from __future__ import annotations
import argparse, os, re, json, string
import random
import pandas as pd
from datasets import Dataset

from unsloth import FastLanguageModel

import torch
from transformers import TrainerCallback
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import wandb

from evaluate import load
bertscore = load("bertscore")

from trl import GRPOTrainer, GRPOConfig

REWARD_WEIGHTS = [0.15, 0.30, 0.25, 0.10, 0.10, 0.10]   # bin, exact, error, desc, len, format

SYSTEM_PROMPT = (
    "You are an expert in evaluating English to Hindi machine translations. Your task is to provide a comprehensive evaluation including a quality score, error categorization, and a detailed analysis.\n\n"
    "**Scoring Guidelines (0-100):**\n"
    "1.\t0-30: Mostly unintelligible - completely inaccurate or containing only some keywords.\n"
    "2.\t31-50: Partial intelligibility - some keywords present but numerous grammatical errors.\n"
    "3.\t51-70: Generally clear - most keywords included with only minor grammatical errors.\n"
    "4.\t71-90: Clear and intelligible - all keywords present with only minor non-grammatical issues.\n"
    "5.\t91-100: Perfect or near-perfect - accurately conveys source meaning without errors.\n\n"

    "**Error Categorization Guidelines:**\n"
    "1.\tUntranslated: A word or phrase in the source is omitted in the translation.\n"
    "2.\tAddition: The translation includes a word or phrase not present in the source.\n"
    "3.\tMistranslation: A word or phrase in the translation does not accurately represent the source meaning.\n"
    "4.\tFluency Error: The translation sounds unnatural due to grammar, spelling, punctuation, or inconsistency.\n"
    "5.\tOther: Any other error not covered by the above categories.\n"
    "6.\tNo Errors: If the translation is perfect.\n\n"

    "Return **exactly** the XML template below (no additional tags):If there are multiple error types, provide them as a comma-separated list inside the <error_type> tag."
    "<reasoning>\n"
    "  <error_type>ERROR_TYPE_1, ERROR_TYPE_2</error_type>\n"
    "  <description>Provide a detailed explanation of the translation errors here.</description>\n"
    "</reasoning>\n"
    "<answer>\n  <da_score>0-100</da_score>\n</answer>"
)

def _prepare_translation_dataset(path: str) -> tuple[Dataset, Dataset, Dataset]:
    df = pd.read_excel(path)
    df["avg_score"] = df["DA_mean"].round().astype(int)

    # comment_cols = [c for c in df.columns if c.lower().startswith("remark")]

    # def _merge(row):
    #     return " ".join(str(row[c]).strip() for c in comment_cols if pd.notna(row[c])) or "No detailed comment provided."

    # df["merged_comment"] = df.apply(_merge, axis=1)
    
    # Use new columns directly
    df["gold_error"] = df["Identified error categories"].fillna("Other")
    df["gold_desc"] = df["Description of the translation"].fillna("No detailed comment provided.")
    
    # Process word-level tags if available
    if "Word-level tags" in df.columns:
        df["word_level_tags"] = df["Word-level tags"].fillna("")
    else:
        df["word_level_tags"] = ""

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    # train_df = train_df[:1000]  # Limit training data for quicker runs; remove or adjust as needed
    # val_df = val_df[:200]       # Limit validation data for quicker runs; remove or adjust as needed

    def _format_word_level_annotation(target_text: str, tags: str) -> str:
        """
        Formats word-level tags in a human-readable way.
        Maps words to their quality tags (OK/BAD) and highlights issues.
        """
        if not tags or not isinstance(tags, str) or tags.strip() == "":
            return "No word-level annotations available."
        
        # Split target text and tags
        words = target_text.split()
        tag_list = tags.strip().split()
        
        # Handle length mismatch gracefully
        if len(words) != len(tag_list):
            # Return basic info if mismatch
            bad_count = tag_list.count("BAD")
            ok_count = tag_list.count("OK")
            total = len(tag_list)
            return f"Word-level quality: {bad_count} problematic words out of {total} total ({ok_count} correct)."
        
        # Create annotated version
        bad_words = []
        annotated_pairs = []
        
        for word, tag in zip(words, tag_list):
            if tag == "BAD":
                bad_words.append(word)
                annotated_pairs.append(f"[{word}:BAD]")
            else:
                annotated_pairs.append(word)
        
        # Build summary
        bad_count = len(bad_words)
        total_words = len(words)
        
        if bad_count == 0:
            return "All words are marked as correct (OK)."
        
        # Create a concise summary
        summary_parts = [
            f"Word-level quality: {bad_count} problematic word(s) out of {total_words}.",
        ]
        
        if bad_count <= 5:
            # If few bad words, list them explicitly
            summary_parts.append(f"Problematic words: {', '.join(bad_words)}.")
        else:
            # If many bad words, give percentage and samples
            percentage = (bad_count / total_words) * 100
            sample_bad = bad_words[:3]
            summary_parts.append(f"Approximately {percentage:.1f}% of words have issues.")
            summary_parts.append(f"Examples of problematic words: {', '.join(sample_bad)}, ...")
        
        return " ".join(summary_parts)

    def _create_dataset_from_df(dataframe: pd.DataFrame) -> Dataset:
        rows = []
        for _, r in dataframe.iterrows():
            # Format word-level annotations
            word_level_info = _format_word_level_annotation(
                r['Target'], 
                r.get('word_level_tags', '')
            )
            
            # Build user content with word-level information
            user_content = (
                f"Source (English): {r['Source']}\n"
                f"Hypothesis (Hindi): {r['Target']}\n\n"
                f"**Word-level Quality Information:**\n{word_level_info}\n\n"
                "Read the source, translated sentence, and word-level quality information above, "
                "then produce your detailed reasoning, error_type, description and DA score."
            )
            
            prompt = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            rows.append({
                "prompt"    : prompt,
                "answer"    : int(r["avg_score"]),
                "gold_error": r["gold_error"],
                "gold_desc" : r["gold_desc"],
            })
        return Dataset.from_list(rows)

    train_dataset = _create_dataset_from_df(train_df)
    eval_dataset = _create_dataset_from_df(val_df)
    
    return train_dataset, eval_dataset

# ─────────────── Reward Calculation Helper ─────────────────────────────────────
SCORE_RE = re.compile(r"<da_score>\s*(\d{1,3})")
ERR_RE   = re.compile(r"<error_type>(.*?)</error_type>", re.S)
DESC_RE  = re.compile(r"<description>(.*?)</description>", re.S)

def _norm(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation)).lower()

def _get_score_bin(score: int | None) -> int | None:
    if score is None:
        return None
    if 1 <= score <= 10:
        return 0
    elif 11 <= score <= 30:
        return 1
    elif 31 <= score <= 50:
        return 2
    elif 51 <= score <= 70:
        return 3
    elif 71 <= score <= 90:
        return 4
    elif 91 <= score <= 100:
        return 5
    return None

def _calculate_error_reward(pred_error: str | None, gold_error_str: str) -> float:
    """Calculates error reward based on Jaccard similarity of error sets."""
    if not pred_error or not gold_error_str:
        return 0.0
    
    # Normalize and split into sets
    gold_errors = set(e.strip().lower() for e in gold_error_str.split(','))
    pred_errors = set(e.strip().lower() for e in pred_error.split(','))

    intersection = len(gold_errors.intersection(pred_errors))
    union = len(gold_errors.union(pred_errors))

    return intersection / union if union > 0 else 0.0

def _calculate_single_generation_rewards_components(
    generated_text: str, 
    gold_score: int, 
    gold_error: str, 
    gold_desc: str
) -> dict:
    """
    Calculates all reward components for a single generated response.
    Returns a dictionary with parsed values (ps, pe, pd) and reward components.
    """
    txt = str(generated_text).strip() if generated_text is not None else ""
    
    ps, pe, pd = None, None, ""
    score_bin_r, score_exact_r, err_r, desc_r, len_r, format_r = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    if not txt: # Handle empty or None generation
        pass # All reward components remain 0.0
    else:
        ps_match = SCORE_RE.search(txt)
        ps = int(ps_match.group(1)) if ps_match else None
        
        pe_match = ERR_RE.search(txt)
        pe = pe_match.group(1).strip() if pe_match else None
        
        pd_match = DESC_RE.search(txt)
        pd = pd_match.group(1).strip() if pd_match else ""

        # Score bin reward
        if ps is not None:
            gold_bin = _get_score_bin(gold_score)
            pred_bin = _get_score_bin(ps)
            score_diff = abs(ps - gold_score)
            
            if gold_bin is not None and pred_bin is not None:
                bin_diff = abs(gold_bin - pred_bin)
                if bin_diff == 0:
                    score_bin_r = 1.0
                elif score_diff <= 5: # Within 5% (5 points) of ground truth
                    score_bin_r = 0.9
                elif bin_diff == 1:
                    score_bin_r = 0.5
                elif bin_diff == 2:
                    if score_diff <= 15:
                         score_bin_r = 0.3
                    else:
                         score_bin_r = 0.0 # Default for 2 buckets away if not close
                else:
                    score_bin_r = 0.0
            elif score_diff <= 5: # If binning fails but score is close
                score_bin_r = 0.9


        # Exact score reward
        if ps is not None:
            score_exact_r = max(0.0, 1.0 - abs(ps - gold_score) / 100.0)
        
        # Error type reward (Jaccard similarity)
        err_r = _calculate_error_reward(pe, gold_error)
        
        # Description reward (BERTScore)
        if pd and gold_desc:
            try:
                # Using F1 score from BERTScore
                _, _, f1 = bertscore.compute(predictions=[pd], 
                                             references=[gold_desc], 
                                             verbose=False,
                                             model_type="xlm-roberta-base")
                desc_r = f1.item()
            except Exception:
                desc_r = 0.0 # Fallback if BERTScore fails
        
        # Length reward - FIXED: count words in description only if it exists
        if pd:  # Only calculate if description was extracted
            n_words = len(pd.split())
            len_r = 1.0 if 60 <= n_words <= 140 else max(0.0, 1 - abs(n_words - 100) / 100.0)
        else:
            len_r = 0.0  # No description = 0 length reward

        # Format reward - FIXED: More lenient, check for all required tags regardless of order
        required_tags = [
            "<reasoning>", "</reasoning>", "<error_type>", "</error_type>",
            "<description>", "</description>", "<answer>", "</answer>",
            "<da_score>", "</da_score>"
        ]
        all_tags_present = all(tag in txt for tag in required_tags)
        content_extracted = ps is not None and pe is not None and pd.strip() != ""

        if all_tags_present and content_extracted:
            format_r = 1.0
        elif all_tags_present or (ps is not None and pe is not None):
            # Give partial credit if tags are present OR if we successfully extracted score and error
            format_r = 0.5
        elif ps is not None or pe is not None or pd.strip() != "":
            # Give minimal credit if we extracted at least something
            format_r = 0.25

    # This combined reward is for logging purpose only.
    # The trainer will use individual rewards if `reward_weights` is set.
    weights = REWARD_WEIGHTS
    components = [score_bin_r, score_exact_r, err_r, desc_r, len_r, format_r]
    if weights and len(weights) == len(components):
        combined_reward_for_logging = sum(w * c for w, c in zip(weights, components))
    else:
        combined_reward_for_logging = sum(components) / len(components) if components else 0.0
    
    return {
        "ps": ps, "pe": pe, "pd": pd,
        "score_bin_r": score_bin_r,
        "score_exact_r": score_exact_r,
        "err_r": err_r,
        "desc_r": desc_r,
        "len_r": len_r,
        "format_r": format_r,
        "combined_reward_qe": combined_reward_for_logging
    }

# ─────────────── Reward Functions for GRPOTrainer ──────────────────────────────

def _create_reward_fn(reward_name: str):
    """Factory to create a reward function for a specific component."""
    def get_reward_component(prompts: list[str] | None = None, completions: list[list[str]] | None = None, **kw) -> list[torch.Tensor]:
        gold_scores = kw.get("answer", [])
        gold_errors = kw.get("gold_error", [])
        gold_descs = kw.get("gold_desc", [])
        all_rewards_tensors: list[torch.Tensor] = []

        if completions is None: return []

        for i in range(len(completions)):
            rewards_for_prompt = []
            for response_text in completions[i]:
                reward_components = _calculate_single_generation_rewards_components(
                    response_text,
                    gold_scores[i] if i < len(gold_scores) else 0,
                    gold_errors[i] if i < len(gold_errors) else "Other",
                    gold_descs[i] if i < len(gold_descs) else ""
                )
                rewards_for_prompt.append(reward_components[reward_name])
            all_rewards_tensors.append(torch.tensor(rewards_for_prompt, dtype=torch.float32))
        return all_rewards_tensors
    return get_reward_component

# Create individual reward functions
score_bin_reward_fn = _create_reward_fn("score_bin_r")
score_exact_reward_fn = _create_reward_fn("score_exact_r")
error_reward_fn = _create_reward_fn("err_r")
description_reward_fn = _create_reward_fn("desc_r")
length_reward_fn = _create_reward_fn("len_r")
format_reward_fn = _create_reward_fn("format_r")


class RoundLogCallback(TrainerCallback):
    """Round float metrics, print sample eval outputs, and log detailed eval to wandb."""
    def __init__(self, tokenizer=None, eval_dataset=None, num_samples_to_print=2, max_completion_length=160, max_wandb_table_samples=20):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset # Store the full dataset
        self.num_samples_to_print = num_samples_to_print # For console
        self.max_wandb_table_samples = max_wandb_table_samples # For wandb.Table
        self.max_completion_length = max_completion_length
        # self.sample_prompts_for_console = [] # Remove pre-selection

        # if eval_dataset and tokenizer and num_samples_to_print > 0:
        #     for i in range(min(num_samples_to_print, len(eval_dataset))):
        #         self.sample_prompts_for_console.append(eval_dataset[i]['prompt'])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None: return
        for k, v in list(logs.items()):
            if isinstance(v, float):
                logs[k] = round(v, 4)
        
        if state.is_world_process_zero and wandb.run is not None:
            wandb.log(logs, step=state.global_step)

    @torch.no_grad()
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        # Console logging for a few samples
        if self.eval_dataset and self.num_samples_to_print > 0 and self.tokenizer and model and state.is_world_process_zero:
            print("\n--- Sample Evaluation Outputs (Console) ---")
            model.eval()

            num_to_sample = min(self.num_samples_to_print, len(self.eval_dataset))
            if num_to_sample > 0:
                random_indices = random.sample(range(len(self.eval_dataset)), num_to_sample)
                
                for i, sample_idx in enumerate(random_indices):
                    sample_data = self.eval_dataset[sample_idx]
                    messages_full_history = sample_data['prompt']
                    c_g_s = sample_data['answer']
                    c_g_e = sample_data['gold_error']
                    c_g_d = sample_data['gold_desc']

                    print(f"\nSample {i+1} (Randomly Selected - Index {sample_idx}):")
                    user_content = "Could not extract user content."
                    for msg_outer_loop_var in reversed(messages_full_history):
                        if msg_outer_loop_var["role"] == "user": user_content = msg_outer_loop_var["content"]; break
                    print(f"Input (User Snippet): \n{user_content}")
                    print(f"Gold Score: {c_g_s}, Gold Error: {c_g_e}, Gold Desc (Snippet): {c_g_d}")

                    messages_for_generation = []
                    for msg_idx, msg_item in enumerate(messages_full_history):
                        if msg_item["role"] == "assistant" and msg_idx == len(messages_full_history) - 1:
                            break 
                        current_content = msg_item["content"]
                        if isinstance(current_content, str):
                            transformed_content = [{"type": "text", "text": current_content}]
                        else:
                            transformed_content = current_content
                        messages_for_generation.append({
                            "role": msg_item["role"],
                            "content": transformed_content
                        })

                    device = model.device
                    tokenized_chat = self.tokenizer.apply_chat_template(
                        messages_for_generation,
                        tokenize=True, add_generation_prompt=True, return_tensors="pt"
                    ).to(device)
                    try:
                        outputs = model.generate(input_ids=tokenized_chat, max_new_tokens=self.max_completion_length, pad_token_id=self.tokenizer.eos_token_id)
                        generated_tokens = outputs[0][tokenized_chat.shape[-1]:]
                        decoded_output = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                        print(f"Generated Output:\n{decoded_output}")

                        reward_components = _calculate_single_generation_rewards_components(
                            decoded_output, c_g_s, c_g_e, c_g_d
                        )
                        
                        ps = reward_components["ps"]
                        pe = reward_components["pe"]
                        pd = reward_components["pd"]
                        c_score_bin_r = reward_components["score_bin_r"]
                        c_score_exact_r = reward_components["score_exact_r"]
                        c_err_r = reward_components["err_r"]
                        c_desc_r = reward_components["desc_r"]
                        c_len_r = reward_components["len_r"]
                        c_format_r = reward_components["format_r"]
                        c_combined_reward_qe = reward_components["combined_reward_qe"]

                        print(f"Pred Score: {ps}, Pred Error: {pe}, Pred Desc (Snippet): {pd}")
                        print(f"Rewards (QE): Bin={c_score_bin_r:.2f}, Exact={c_score_exact_r:.2f}, Err={c_err_r:.2f}, Desc={c_desc_r:.2f}, Len={c_len_r:.2f}, Fmt={c_format_r:.2f} => Combined_QE={c_combined_reward_qe:.2f}")

                    except Exception as e:
                        print(f"Error during console sample generation or reward calculation: {e}")
            print("--- End Sample Evaluation Outputs (Console) ---\n")

        # WandB logging for evaluation set
        if state.is_world_process_zero and wandb.run is not None and self.eval_dataset and self.tokenizer and model:
            print("\n--- Generating Detailed Evaluation Metrics for WandB (on a subset of eval data) ---")
            model.eval()
            
            all_pred_scores, all_gold_scores = [], []
            all_score_exact_r_list, all_err_r_list, all_desc_r_list, all_len_r_list = [], [], [], []
            all_score_bin_r_list, all_format_r_list, all_combined_reward_qe_list = [], [], []
            
            eval_table = wandb.Table(columns=[
                "Input (User)", "Gold Score", "Gold Error", "Gold Desc Snippet", 
                "Generated Output", "Pred Score", "Pred Error", "Pred Desc Snippet",
                "ScoreExact_R", "Err_R", "Desc_R", "Len_R", 
                "ScoreBin_R", "Format_R", "Combined_R_QE"
            ])

            num_samples_for_wandb_detail = min(self.max_wandb_table_samples, len(self.eval_dataset))

            for idx in range(num_samples_for_wandb_detail): 
                item = self.eval_dataset[idx] 
                messages_full_history = item['prompt'] 
                g_s = item['answer']
                g_e = item['gold_error']
                g_d = item['gold_desc']

                user_content = "N/A"
                for msg in reversed(messages_full_history):
                    if msg["role"] == "user": user_content = msg["content"]; break

                messages_for_generation = []
                for msg_idx, msg_item in enumerate(messages_full_history): # Changed variable names for consistency
                    if msg_item["role"] == "assistant" and msg_idx == len(messages_full_history) - 1:
                        break 
                    current_content = msg_item["content"]
                    if isinstance(current_content, str):
                        transformed_content = [{"type": "text", "text": current_content}]
                    else:
                        transformed_content = current_content
                    messages_for_generation.append({
                        "role": msg_item["role"],
                        "content": transformed_content
                    })

                tokenized_chat = self.tokenizer.apply_chat_template(
                    messages_for_generation, 
                    tokenize=True, add_generation_prompt=True, return_tensors="pt"
                ).to(model.device)
                
                generated_output_text = ""
                reward_components = {} # Initialize

                try:
                    outputs = model.generate(input_ids=tokenized_chat, max_new_tokens=self.max_completion_length, pad_token_id=self.tokenizer.eos_token_id)
                    generated_tokens = outputs[0][tokenized_chat.shape[-1]:]
                    generated_output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

                    reward_components = _calculate_single_generation_rewards_components(
                        generated_output_text, g_s, g_e, g_d
                    )

                except Exception as e:
                    print(f"Error during wandb sample generation or reward calculation for item {idx}: {e}")
                    # Ensure default values if error occurs before reward_components is populated
                    reward_components = _calculate_single_generation_rewards_components("", g_s, g_e, g_d)


                ps = reward_components.get("ps")
                pe = reward_components.get("pe")
                pd = reward_components.get("pd", "") # Default to empty string if not found
                score_exact_r = reward_components.get("score_exact_r", 0.0)
                err_r = reward_components.get("err_r", 0.0)
                desc_r = reward_components.get("desc_r", 0.0)
                len_r = reward_components.get("len_r", 0.0)
                score_bin_r = reward_components.get("score_bin_r", 0.0)
                format_r = reward_components.get("format_r", 0.0)
                combined_reward_qe = reward_components.get("combined_reward_qe", 0.0)

                all_gold_scores.append(g_s)
                if ps is not None: all_pred_scores.append(ps)
                
                all_score_exact_r_list.append(score_exact_r)
                all_err_r_list.append(err_r)
                all_desc_r_list.append(desc_r)
                all_len_r_list.append(len_r)
                
                all_score_bin_r_list.append(score_bin_r)
                all_format_r_list.append(format_r)
                all_combined_reward_qe_list.append(combined_reward_qe)

                eval_table.add_data(
                    user_content, g_s, g_e, g_d,
                    generated_output_text, ps, pe, pd,
                    round(score_exact_r, 4), round(err_r, 4), round(desc_r, 4), round(len_r, 4),
                    round(score_bin_r, 4), round(format_r, 4), round(combined_reward_qe, 4)
                )
            
            metrics_to_log = {"eval/eval_samples_table": eval_table}
            if all_pred_scores: 
                valid_gold_scores = [all_gold_scores[i] for i, score in enumerate(all_pred_scores) if score is not None] 
                valid_pred_scores = [s for s in all_pred_scores if s is not None]
                if valid_pred_scores: 
                    metrics_to_log["eval/DA_score_MSE_subset"] = mean_squared_error(valid_gold_scores, valid_pred_scores)
                    metrics_to_log["eval/DA_score_MAE_subset"] = mean_absolute_error(valid_gold_scores, valid_pred_scores)
                    metrics_to_log["eval/DA_score_RMSE_subset"] = np.sqrt(metrics_to_log["eval/DA_score_MSE_subset"])
            
            # Log means of qe_reward components
            metrics_to_log["eval/mean_score_exact_reward_subset"] = np.mean(all_score_exact_r_list) if all_score_exact_r_list else 0
            metrics_to_log["eval/mean_error_reward_subset"] = np.mean(all_err_r_list) if all_err_r_list else 0
            metrics_to_log["eval/mean_description_reward_subset"] = np.mean(all_desc_r_list) if all_desc_r_list else 0
            metrics_to_log["eval/mean_length_reward_subset"] = np.mean(all_len_r_list) if all_len_r_list else 0
            metrics_to_log["eval/mean_score_bin_reward_subset"] = np.mean(all_score_bin_r_list) if all_score_bin_r_list else 0
            metrics_to_log["eval/mean_format_reward_subset"] = np.mean(all_format_r_list) if all_format_r_list else 0
            metrics_to_log["eval/mean_combined_reward_qe_subset"] = np.mean(all_combined_reward_qe_list) if all_combined_reward_qe_list else 0

            wandb.log(metrics_to_log, step=state.global_step)
            print("--- Finished Detailed Evaluation Metrics for WandB (on subset) ---")

class FixedGRPOTrainer(GRPOTrainer):
    """Custom GRPO Trainer that fixes the _get_train_sampler signature issue."""
    
    def _get_train_sampler(self, *args, **kwargs):
        """Override to handle dataset argument properly."""
        # If called with dataset argument, ignore it and use self.train_dataset
        return super()._get_train_sampler()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Train_Deepseek.xlsx")
    ap.add_argument("--output_dir", default="grpo_out")
    ap.add_argument("--max_steps", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps.")
    ap.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for training.")
    ap.add_argument("--max_completion_length", type=int, default=160, help="Max length of generated completions.")
    ap.add_argument("--num_generations", type=int, default=4, help="Number of generations per prompt.")
    ap.add_argument("--eval_steps", type=int, default=10, help="Frequency of evaluation in steps.")
    ap.add_argument("--wandb_project", type=str, default="rlqe", help="WandB project name.")
    args = ap.parse_args()

    # Initialize WandB
    wandb.init(project=args.wandb_project, config=vars(args))

    train_ds, eval_ds = _prepare_translation_dataset(args.data)
    print(f"Loaded {len(train_ds)} training examples, {len(eval_ds)} validation examples.")

    model, tok = FastLanguageModel.from_pretrained(
        "unsloth/gemma-3-4b-it-bnb-4bit", max_seq_length=2048,
        load_in_4bit=True, device_map="auto")
    model = FastLanguageModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True, 
        r=32, 
        lora_alpha=64, 
        lora_dropout=0.05, 
        bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],)

    actual_eval_steps = max(1, min(args.eval_steps, args.max_steps))
    if args.eval_steps > args.max_steps:
        print(f"Warning: eval_steps ({args.eval_steps}) > max_steps ({args.max_steps}). Setting eval_steps to {args.max_steps}.")
    if args.eval_steps <= 0:
        print(f"Warning: eval_steps ({args.eval_steps}) is not positive. Setting eval_steps to 1.")

    reward_weights = REWARD_WEIGHTS 

    cfg = GRPOConfig(
        output_dir=args.output_dir, 
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size, # Set eval batch size same as train
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        num_generations=args.num_generations, 
        max_completion_length=args.max_completion_length,
        remove_unused_columns=False,
        reward_weights=reward_weights, # List of weights for each reward function (comment for equal weightage)
        fp16=torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8,
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8,
        log_level="info",
        logging_strategy="steps", 
        logging_steps=1,
        # importance_sampling_level="token",
        # scale_rewards=False,
        # loss_type="dr_grpo",
        save_steps=actual_eval_steps,          
        eval_strategy="steps",
        eval_steps=actual_eval_steps,
        report_to="wandb",
    )
    
    # Log GRPOConfig parameters to wandb
    wandb.config.update({
        "grpo_output_dir": cfg.output_dir,
        "grpo_per_device_train_batch_size": cfg.per_device_train_batch_size,
        "grpo_per_device_eval_batch_size": cfg.per_device_eval_batch_size, 
        "grpo_gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "grpo_learning_rate": cfg.learning_rate,
        "grpo_max_steps": cfg.max_steps,
        "grpo_num_generations": cfg.num_generations,
        "grpo_max_completion_length": cfg.max_completion_length,
        "grpo_fp16": cfg.fp16,
        "grpo_bf16": cfg.bf16,
        "grpo_logging_strategy": cfg.logging_strategy,
        "grpo_logging_steps": cfg.logging_steps, 
        "grpo_eval_strategy": cfg.eval_strategy,
        "grpo_eval_steps": cfg.eval_steps,
        "grpo_reward_weights": cfg.reward_weights, # Log the weights
    })
    
    num_samples_for_console_print = 2 
    
    round_log_callback = RoundLogCallback(
        tokenizer=tok, 
        eval_dataset=eval_ds, 
        num_samples_to_print=num_samples_for_console_print,
        max_completion_length=cfg.max_completion_length, 
        max_wandb_table_samples=5 
    )

    # The trainer will use the list of reward functions and combine them using the weights from GRPOConfig.
    reward_fns = [
        score_bin_reward_fn,
        score_exact_reward_fn,
        error_reward_fn,
        description_reward_fn,
        length_reward_fn,
        format_reward_fn,
    ]

    trainer = FixedGRPOTrainer(  # Changed from GRPOTrainer to FixedGRPOTrainer
        model=model, 
        tokenizer=tok, 
        reward_funcs=reward_fns, # List of reward functions
        args=cfg,
        train_dataset=train_ds, 
        eval_dataset=eval_ds, 
        callbacks=[round_log_callback]
    )
    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    lora_dir = os.path.join(args.output_dir, "lora")
    try:
        model.save_lora(lora_dir)
    except AttributeError:
        model.save_pretrained(lora_dir)
    print(f"LoRA adapter saved to {lora_dir}")
    
    print(f"\nTraining complete. Model artifacts saved to {args.output_dir}.")

    # Save merged model (optional, can be done after test evaluation)
    merged_dir = os.path.join(args.output_dir, "merged_16bit")
    try:
        print(f"Attempting to save merged model to {merged_dir}...")
        model.save_pretrained_merged(merged_dir, tok, save_method="merged_16bit")
        print(f"Merged model saved to {merged_dir}")
    except Exception as e:
        print(f"Could not save merged model: {e}. The LoRA adapter is saved in {lora_dir}.")
    
    wandb.finish()

if __name__ == "__main__":
    main()
