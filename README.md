# Fine-Tuner for Classifier
A simple, efficient script for fine-tuning a Hugging Face text classification model.  
**by Anas**

This repository contains a Python script (`finetune_classifier.py`) that allows you to:

- Load a **Hugging Face model** (e.g., `distilbert-base-uncased`).
- Load a **dataset** (local CSV/JSON or a public dataset from the Hugging Face Hub).
- Automatically map labels (e.g., `negative`, `positive`) for multi-class classification.
- Fine-tune the model with adjustable **hyperparameters**.
- Save the resulting **fine-tuned model** and tokenizer in a specified output directory.

---

## 1. Dependencies

- **Python 3.8+** is recommended
- **PyTorch** (CPU or GPU version)
- **Transformers**, **Datasets**, and **Accelerate** (Hugging Face libraries)

### Quick Installation

```bash
pip install --upgrade torch       # For CPU only, or:
# pip install torch --index-url https://download.pytorch.org/whl/cu118  # For CUDA support
pip install --upgrade transformers datasets accelerate
```

## 2. Project Files

- **`finetune_classifier.py`**: The main script for training and fine-tuning.
- **`dataset.csv`** (example): A sample dataset with `text` and `label` columns.

## 3. Usage

Run the script with the following required arguments:

- `--model_name`: Model name or path (e.g., `distilbert-base-uncased`).
- `--dataset_name`: Path to your local `.csv`/`.json` file or the name of a Hugging Face dataset.
- `--text_column` and `--label_column`: Column names for text and labels.
- `--var_predict`: Space-separated list of possible labels (e.g., `negative positive`).

### Optional Arguments:

- `--output_dir`: Output directory (default: `./output_clf`).
- `--num_train_epochs`: Number of epochs (default: 3).
- `--fp16`: Use half-precision if available (recommended for GPU).
- `--eval_strategy` (or `--evaluation_strategy`): Evaluation strategy (`no`, `steps`, or `epoch`).

### Example Command (Windows PowerShell)

```powershell
python finetune_classifier.py `
  --model_name distilbert-base-uncased `
  --dataset_name dataset.csv `
  --text_column text `
  --label_column label `
  --var_predict negative positive `
  --output_dir ./model_clf `
  --num_train_epochs 3 `
  --fp16


## 4. Output

After training, the script will generate:

1. A folder (e.g., `./model_clf`) containing the fine-tuned model weights.
2. A tokenizer configuration.

You can reload this model using:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("./model_clf")
tokenizer = AutoTokenizer.from_pretrained("./model_clf")
```

## 5. Notes & Tips

1. **Library Versions**: Ensure that `transformers`, `datasets`, `accelerate`, and `torch` versions are compatible to avoid warnings or errors.
2. **Multi-class Classification**: To handle more than two labels, list them all under `--var_predict` (e.g., `label1 label2 label3`).
3. **Data Privacy**: Anonymize sensitive data if required by regulations.
4. **GPU Usage**: Install the CUDA version of PyTorch for faster training and use `--fp16` for better performance.

## Enjoy Fine-Tuning!

If you encounter issues or have suggestions, feel free to create an issue in this repository or reach out directly.

Happy fine-tuning!
