# ALOPE-RL

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Unsloth](https://img.shields.io/badge/Powered%20by-Unsloth-orange.svg)](https://github.com/unslothai/unsloth)

ALOPE-RL is a policy-based reinforcement learning framework for **Machine Translation Quality Estimation (QE)**. It leverages the **GRPO (Group Relative Policy Optimization)** algorithm to train efficient adapters for Large Language Models (LLMs), enabling them to generate precise quality scores, error categorizations, and detailed **Translation Quality Remarks (TQR)**.

The framework is specifically designed to address gaps in low-resource language evaluation (e.g., English -> Malayalam, English -> Hindi) by utilizing rewards derived from Direct Assessment (DA) scores and contextual annotator comments.

---

## 🚀 Key Features

- **Policy-Based Reinforcement Learning**: Implements the **GRPO** algorithm via the `trl` library, enabling high-performance policy optimization without the overhead of a separate critic model.
- **TQR-Augmented Training**: Leverages **Translation Quality Remarks (TQR)** -- contextual annotator comments to drive better judgment and explainability in QE outputs.
- **Multi-Component Reward System**: Models are optimized using a weighted reward aggregation system:
  - **DA Score Accuracy**: Proximity to ground-truth Direct Assessment (DA) scores (Exact Score & Score Bin).
  - **Error Categorization**: Accuracy in identifying specific error types (e.g., Mistranslation, Addition, Untranslated).
  - **Description Quality**: Semantic similarity of generated TQR via **BERTScore**.
  - **Formatting & Length**: Ensures output adheres to a strict XML structure and maintains optimal verbosity.
- **Efficient Fine-Tuning**: Built on **Unsloth**, utilizing 4-bit quantization and LoRA adapters to achieve state-of-the-art results with compact LLMs (≤4B parameters).

---

## 📂 Repository Structure

| File | Description |
| :--- | :--- |
| `rlqe_word_tags.py` | Training script utilizing word-level quality tags (OK/BAD) for English-Hindi evaluation. |
| `rlqe_weak_annotations.py` | Training script utilizing Translation Quality Remarks (TQR) for English-Malayalam evaluation. |
| `evaluate_word_tags.py` | Evaluation script for word-tag models, calculating MSE, MAE, and correlation metrics. |
| `evaluate_weak_annotations.py` | Evaluation script for TQR-based models. |
| `rlqe_yaml.yml` | Full Conda environment specification. |
| `requirements.txt` | Standard pip dependency list. |
| `LICENSE` | MIT License. |

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/surrey-nlp/ALOPE-RL.git
cd ALOPE-RL
```

### 2. Set Up Environment
We recommend using Conda:
```bash
conda env create -f rlqe_yaml.yml
conda activate rlqe
```
Alternatively:
```bash
pip install -r requirements.txt
```

---

## 🚄 Usage

### Training with Translation Quality Remarks (TQR)
To start training a model using GRPO and contextual annotations:
```bash
python rlqe_weak_annotations.py --data your_data.xlsx --max_steps 100 --batch_size 64
```

### Training with Word-Level Tags
To train using word-level `OK`/`BAD` tags as additional context:
```bash
python rlqe_word_tags.py --data your_data.xlsx --max_steps 100 --batch_size 64
```

### Evaluation
Evaluate model performance against a test set to get MSE, MAE, and correlation coefficients:

For TQR-based models:
```bash
python evaluate_weak_annotations.py
```

For Word-Tag models:
```bash
python evaluate_word_tags.py
```

---

## 🧠 Model Architecture

The ALOPE-RL architecture consists of:
- **QE Model**: A frozen Large Language Model (e.g., Gemma-3-4b-it) augmented with trainable **LoRA** adapters.
- **GRPO Trainer**: Manages terminal rewards and weight updates across a group of `K` completions per prompt.
- **Reward Aggregation**: A weighted sum of policy rewards derived from both scalar scores and linguistic analysis.

---

## 📊 Reward Function Details

The GRPO trainer optimizes the model based on the following weighted reward components:

1.  **Exact Score Reward (30%)**: Numerical proximity to the gold DA score.
2.  **Error Type Reward (25%)**: Jaccard similarity between predicted and gold error categories.
3.  **Score Bin Reward (15%)**: Correctly identifying the 0-100 score bucket.
4.  **Description Reward (10%)**: BERTScore F1 between generated TQR and gold remarks.
5.  **Length Reward (10%)**: Ensuring the description length is within the expected range.
6.  **Format Reward (10%)**: Strict adherence to the required XML schema (`<reasoning>`, `<answer>`, etc.).

We can customize the weights and the rewards.
---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
