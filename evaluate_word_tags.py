import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr, kendalltau
from unsloth import FastLanguageModel
import torch
import re
from peft import PeftModel
from tqdm import tqdm

# Configuration
EXCEL_FILE_PATH = "en-hi_TEST_combined.xlsx"
ORIGINAL_TEXT_COLUMN = 'Source'
TRANSLATED_TEXT_COLUMN = 'Target'
# WEAK_ANNOTATIONS_COLUMN_1 = 'Remark-A1'
# WEAK_ANNOTATIONS_COLUMN_2 = 'Remark-A2'
# WEAK_ANNOTATIONS_COLUMN_3 = 'Remark-A3'
WORD_LEVEL_TAGS_COLUMN = 'Word-level tags'  # New column for word-level quality tags
DA_SCORE_COLUMN = 'DA_mean' # Ground truth

# Model Paths
OFF_THE_SHELF_MODEL_NAME = "unsloth/gemma-3-4b-it-bnb-4bit"
LORA_ADAPTER_PATH = "grpo_out/lora"
# LORA_ADAPTER_PATH_ALTERNATIVE = "grpo_out/merged_16bit"

MAX_SEQ_LENGTH = 2048
DTYPE = None
LOAD_IN_4BIT = False # Use 4bit quantization to reduce memory usage

def load_and_split_data(file_path):
    """Loads data from Excel and splits it into train, validation, and test sets."""
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None, None, None

    # Combine weak annotation columns, handling potential NaN values
    # df['weak_annotations'] = df[[WEAK_ANNOTATIONS_COLUMN_1, WEAK_ANNOTATIONS_COLUMN_2, WEAK_ANNOTATIONS_COLUMN_3]].apply(
    #     lambda x: ' '.join(x.dropna().astype(str)), axis=1
    # )

    required_columns = [ORIGINAL_TEXT_COLUMN, TRANSLATED_TEXT_COLUMN, DA_SCORE_COLUMN]
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Required column '{col}' not found in the Excel file.")
            print(f"Available columns: {df.columns.tolist()}")
            return None, None, None

    # Drop rows where DA_SCORE_COLUMN is NaN, as it's the target
    df.dropna(subset=[DA_SCORE_COLUMN], inplace=True)
    
    # Fill NaN in text fields with empty strings
    df[ORIGINAL_TEXT_COLUMN] = df[ORIGINAL_TEXT_COLUMN].fillna('')
    df[TRANSLATED_TEXT_COLUMN] = df[TRANSLATED_TEXT_COLUMN].fillna('')
    
    # Process word-level tags if available
    if WORD_LEVEL_TAGS_COLUMN in df.columns:
        df[WORD_LEVEL_TAGS_COLUMN] = df[WORD_LEVEL_TAGS_COLUMN].fillna('')
    else:
        df[WORD_LEVEL_TAGS_COLUMN] = ''
        print(f"Warning: '{WORD_LEVEL_TAGS_COLUMN}' column not found. Proceeding without word-level annotations.")

    print(f"Test samples: {len(df)} ({(len(df)/len(df)*100):.2f}%)")
    
    return df

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

def format_prompt_for_da_score(original_text, translated_text, word_level_tags=""):
    """Formats the input for the model to predict a DA score."""
    word_level_info = _format_word_level_annotation(translated_text, word_level_tags)
    
    prompt = f"""<s>[INST] You are an expert evaluator of machine translation quality.
Your task is to predict a Direct Assessment (DA) score for a given translation.
The DA score should be a single floating-point number between 0 and 100, where 0 indicates a completely nonsensical translation and 100 indicates a perfect translation.
Consider the original source text, the machine-translated text, and any available weak annotations (such as comments or error tags) to make your prediction.

Original Text:
{original_text}

Translated Text:
{translated_text}

**Word-level Quality Information:**
{word_level_info}

Based on all the provided information, what is the DA score for the translation?
Provide only the numerical score. [/INST]
DA Score: """
    return prompt

def format_prompt_for_da_score_rlqe(original_text, translated_text, word_level_tags=""):
    """RLQE-style prompt format combining error info and DA score."""
    word_level_info = _format_word_level_annotation(translated_text, word_level_tags)
    
    prompt = f"""<s>[INST]You are an expert in evaluating English to Hindi machine translations. Your task is to provide a comprehensive evaluation including a quality score, error categorization, and a detailed analysis.

**Scoring Guidelines (0-100):**
1. 0-30: Mostly unintelligible - completely inaccurate or containing only some keywords.
2. 31-50: Partial intelligibility - some keywords present but numerous grammatical errors.
3. 51-70: Generally clear - most keywords included with only minor grammatical errors.
4. 71-90: Clear and intelligible - all keywords present with only minor non-grammatical issues.
5. 91-100: Perfect or near-perfect - accurately conveys source meaning without errors.

**Error Categorization Guidelines:**
1. Untranslated: A word or phrase in the source is omitted in the translation.
2. Addition: The translation includes a word or phrase not present in the source.
3. Mistranslation: A word or phrase in the translation does not accurately represent the source meaning.
4. Fluency Error: The translation sounds unnatural due to grammar, spelling, punctuation, or inconsistency.
5. Other: Any other error not covered by the above categories.
6. No Errors: If the translation is perfect.

Return **exactly** the XML template below (no additional tags). If there are multiple error types, provide them as a comma-separated list inside the <error_type> tag.
<reasoning>
  <error_type>ERROR_TYPE_1, ERROR_TYPE_2</error_type>
  <description>Provide a detailed explanation of the translation errors here.</description>
</reasoning>
<answer>
  <da_score>0-100</da_score>
</answer>

Source (English): {original_text}
Hypothesis (Hindi): {translated_text}

**Word-level Quality Information:**
{word_level_info}

Read the source, translated sentence, and word-level quality information above, then produce your detailed reasoning, error_type, description, and final DA score. [/INST]
"""
    return prompt

def get_model_prediction(model, tokenizer, prompt_text, use_rlqe_prompt=False):
    """Generates a prediction from the model for a given prompt."""
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH).to("cuda")
    input_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        if hasattr(model, "current_lora"):
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,  # Increased to capture complete XML
                pad_token_id=tokenizer.eos_token_id,
                lora_request=model.current_lora,
            )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,  # Increased to capture complete XML
                pad_token_id=tokenizer.eos_token_id,
            )
    
    generated_ids = outputs[0, input_length:]
    decoded_generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print(f"Raw generated text: '{decoded_generated_text}'") 

    try:
        # \d+\.?\d*  -> one or more digits, optionally followed by a dot and zero or more digits (e.g., "75", "75.0", "75.")
        # |\.\d+     -> OR a dot followed by one or more digits (e.g., ".5")
        if use_rlqe_prompt:
            match = re.search(r"<da_score>\s*(\d+\.?\d*|\.\d+)\s*</da_score>", decoded_generated_text, re.IGNORECASE)
        else:
            match = re.search(r"(\d+\.?\d*|\.\d+)", decoded_generated_text)
        
        if match:
            score_str = match.group(1)
            try:
                parsed_score = float(score_str)
                if 0 <= parsed_score <= 100:
                    return parsed_score
                else:
                    print(f"Warning: Parsed score {parsed_score} is outside the expected 0-100 range. From generated: '{decoded_generated_text}'. Returning 0.0 as a fallback.")
                    return 0.0
            except ValueError:
                print(f"Warning: ValueError converting '{score_str}' to float. From generated: '{decoded_generated_text}'. Returning 0.0.")
                return 0.0
        else:
            print(f"Warning: Could not parse a valid score from model generated text: '{decoded_generated_text}'. Returning 0.0.")
            return 0.0

    except Exception as e:
        print(f"Warning: General error parsing score from generated text '{decoded_generated_text}'. Error: {e}. Returning 0.0.")
        return 0.0

def run_evaluation_on_test_set(model, tokenizer, test_df, use_rlqe_prompt=False):
    """Runs evaluation on the test set and prints metrics."""
    if test_df is None or test_df.empty:
        print("Test dataframe is empty or None. Skipping evaluation.")
        return

    predictions = []
    ground_truths = []

    print(f"\nRunning evaluation on {len(test_df)} test samples...")
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
        original = row[ORIGINAL_TEXT_COLUMN]
        translation = row[TRANSLATED_TEXT_COLUMN]
        word_level_tags = row.get(WORD_LEVEL_TAGS_COLUMN, "")
        true_da_score = row[DA_SCORE_COLUMN]

        if use_rlqe_prompt:
            prompt = format_prompt_for_da_score_rlqe(original, translation, word_level_tags)
            predicted_da_score = get_model_prediction(model, tokenizer, prompt, use_rlqe_prompt=True)
        else:
            prompt = format_prompt_for_da_score(original, translation, word_level_tags)
            predicted_da_score = get_model_prediction(model, tokenizer, prompt, use_rlqe_prompt=False)
        
        predictions.append(predicted_da_score)
        ground_truths.append(true_da_score)

    if not predictions or not ground_truths:
        print("No predictions or ground truths generated. Cannot calculate metrics.")
        return

    mse = mean_squared_error(ground_truths, predictions)
    mae = mean_absolute_error(ground_truths, predictions)
    pearson_corr, pearson_pval = pearsonr(ground_truths, predictions)
    spearman_corr, spearman_pval = spearmanr(ground_truths, predictions)
    kendall_corr, kendall_pval = kendalltau(ground_truths, predictions)

    print("\nEvaluation Metrics:")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  Pearson Correlation: {pearson_corr:.4f} (p-value: {pearson_pval:.4e})")
    print(f"  Spearman Correlation: {spearman_corr:.4f} (p-value: {spearman_pval:.4e})")
    print(f"  Kendall's Tau: {kendall_corr:.4f} (p-value: {kendall_pval:.4e})")

def evaluate_off_the_shelf_gemma_model(test_df):
    """Evaluates the off-the-shelf Gemma model."""
    print(f"\n--- Evaluating Off-the-Shelf Gemma Model: {OFF_THE_SHELF_MODEL_NAME} ---")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=OFF_THE_SHELF_MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    model.eval() # Set model to evaluation mode

    run_evaluation_on_test_set(model, tokenizer, test_df, use_rlqe_prompt=True)

def evaluate_lora_trained_gemma_model(test_df, adapter_path):
    """Evaluates the LoRA trained Gemma model."""
    print(f"\n--- Evaluating LoRA Trained Gemma Model ---")
    print(f"Loading base model: {OFF_THE_SHELF_MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=OFF_THE_SHELF_MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    try:
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
    except Exception as e:
        print(f"Error loading LoRA adapter from {adapter_path}: {e}")
        print("Skipping evaluation for this LoRA model.")
        return

    run_evaluation_on_test_set(model, tokenizer, test_df, use_rlqe_prompt=True)

if __name__ == "__main__":
    print("Starting evaluation script...")
    test_data = load_and_split_data(EXCEL_FILE_PATH)

    if test_data is not None and not test_data.empty:
        # Evaluate the off-the-shelf model
        # evaluate_off_the_shelf_gemma_model(test_data.copy()) # Pass a copy to avoid modifications

        # Evaluate the LoRA trained model
        evaluate_lora_trained_gemma_model(test_data.copy(), LORA_ADAPTER_PATH)

    else:
        print("No test data available to run evaluations.")
    
    print("\nEvaluation script finished.")
