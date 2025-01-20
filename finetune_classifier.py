import argparse
import os
import numpy as np

from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a text classifier with custom labels."
    )
    # --- Paramètres de base ---
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased",
                        help="Chemin ou nom du modèle Hugging Face.")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Chemin du fichier CSV/JSON ou nom d'un dataset HF.")
    parser.add_argument("--text_column", type=str, default="text",
                        help="Colonne contenant le texte.")
    parser.add_argument("--label_column", type=str, default="label",
                        help="Colonne contenant le label.")
    parser.add_argument("--var_predict", nargs="+", required=True,
                        help="Liste des labels possibles, ex: negative positive.")
    
    # --- Training args ---
    parser.add_argument("--output_dir", type=str, default="./output_clf",
                        help="Répertoire où sauvegarder le modèle fine-tuné.")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Nombre d'époques.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8,
                        help="Taille de batch en entraînement.")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Taille maximale de séquence.")
    parser.add_argument("--fp16", action="store_true",
                        help="Utiliser le FP16 si disponible (GPU).")

    # On remplace l'ancien `evaluation_strategy` par `eval_strategy`
    parser.add_argument("--eval_strategy", type=str, default="no",
                        choices=["no", "epoch", "steps"],
                        help="Stratégie d'évaluation (no, steps, epoch).")

    args = parser.parse_args()
    
    # -----------------------
    # 1. Préparer la liste des labels (var_predict)
    # -----------------------
    label_list = args.var_predict  # ex: ["negative", "positive"]
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}
    num_labels = len(label_list)
    
    # -----------------------
    # 2. Charger le modèle et le tokenizer
    # -----------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    
    # -----------------------
    # 3. Charger le dataset
    # -----------------------
    if os.path.isfile(args.dataset_name):
        ext = args.dataset_name.split(".")[-1]
        if ext == "csv":
            raw_dataset = load_dataset("csv", data_files=args.dataset_name)
        elif ext == "json":
            raw_dataset = load_dataset("json", data_files=args.dataset_name)
        else:
            raise ValueError("Format non géré: utilisez .csv ou .json.")
    else:
        raw_dataset = load_dataset(args.dataset_name)
    
    if "train" in raw_dataset:
        train_dataset = raw_dataset["train"]
    else:
        train_dataset = raw_dataset[list(raw_dataset.keys())[0]]
    
    if "test" in raw_dataset:
        eval_dataset = raw_dataset["test"]
    elif "validation" in raw_dataset:
        eval_dataset = raw_dataset["validation"]
    else:
        eval_dataset = None
    
    # -----------------------
    # 4. Préparer la tokenization
    # -----------------------
    def tokenize_fn(examples):
        return tokenizer(
            examples[args.text_column],
            truncation=True,
            max_length=args.max_length
        )
    
    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    if eval_dataset:
        eval_dataset = eval_dataset.map(tokenize_fn, batched=True)
    
    def label_to_id(example):
        label_str = example[args.label_column]
        return {"labels": label2id[label_str]}
    
    train_dataset = train_dataset.map(label_to_id)
    if eval_dataset:
        eval_dataset = eval_dataset.map(label_to_id)
    
    keep_cols = {"input_ids", "attention_mask", "labels"}
    remove_cols = [col for col in train_dataset.column_names if col not in keep_cols]
    train_dataset = train_dataset.remove_columns(remove_cols)
    if eval_dataset:
        eval_dataset = eval_dataset.remove_columns(remove_cols)
    
    # -----------------------
    # 5. Préparer Trainer et TrainingArguments
    # -----------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        # On utilise `eval_strategy` à la place de `evaluation_strategy`
        eval_strategy=args.eval_strategy,
        fp16=args.fp16,
        save_steps=500,
        logging_steps=50
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        accuracy = np.mean(preds == labels)
        return {"accuracy": accuracy}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if eval_dataset else None,
        data_collator=data_collator,
        compute_metrics=compute_metrics if eval_dataset else None,
    )
    
    # -----------------------
    # 6. Entraînement
    # -----------------------
    trainer.train()
    
    # -----------------------
    # 7. Sauvegarde
    # -----------------------
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Modèle et tokenizer sauvegardés dans {args.output_dir}")


if __name__ == "__main__":
    main()
