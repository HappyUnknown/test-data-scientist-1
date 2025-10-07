import json
import argparse
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification 
)
import torch
from typing import List, Dict, Any
import sys
import os # Added os for checking output directory

# Define the custom tags used in the dataset generator
LABELS: List[str] = ["O", "B-ANIMAL", "I-ANIMAL"]
ID2LABEL: Dict[int, str] = {i: label for i, label in enumerate(LABELS)}
LABEL2ID: Dict[str, int] = {label: i for i, label in enumerate(LABELS)}

def tokenize_and_align_labels(examples: Dict[str, Any], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    """
    Tokenizes text and aligns NER tags with new sub-tokens.
    Special tokens (CLS, SEP) and subsequent sub-tokens of a word get -100 label, 
    which is ignored by the Hugging Face loss function.
    """
    # Tokenize the words, preserving word boundaries
    tokenized_inputs = tokenizer(
        examples["tokens"], 
        truncation=True, 
        is_split_into_words=True,
        # Use an appropriate maximum length if needed, e.g., max_length=128
    )
    
    labels = []
    
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # 1. Handle Special Tokens (e.g., [CLS], [SEP])
            if word_idx is None:
                label_ids.append(-100)
            # 2. Handle the first sub-token of a new word
            elif word_idx != previous_word_idx:
                # The label is retrieved using the original word index
                label_ids.append(label[word_idx])
            # 3. Handle subsequent sub-tokens of the same word
            else:
                # Assign the same label as the start of the word. 
                label_ids.append(label[word_idx]) 
            
            previous_word_idx = word_idx
        labels.append(label_ids)
        
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a BERT model for Animal NER.")
    
    # Flagged arguments with defaults
    parser.add_argument("--model_name", type=str, default="bert-base-cased", help="Base transformer model for fine-tuning.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--test_size", type=float, default=0.1, help="Fraction of data to use for evaluation.")

    # Positional arguments (These match your execution style: python script arg1 arg2)
    # 1. Positional argument for Input Data Path
    parser.add_argument(
        "input_file", 
        type=str,
        help="The path to the training data JSON file (e.g., '../dataset/ner_animal_model/ner_training_data_enhanced.json')."
    )

    # 2. Positional argument for Output Directory Path
    parser.add_argument(
        "output_directory", 
        type=str,
        help="The directory where the trained model will be saved (e.g., '../dataset/ner_animal_model')."
    )
    
    args = parser.parse_args() 

    # Check for GPU availability
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU. Training will be slower.")

    # 1. Load Data
    # FIX: Access the positional argument using args.input_file
    print(f"Loading data from {args.input_file}...")
    try:
        with open(args.input_file, 'r') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Training data {args.input_file} not found. Did you run ner_data_generator.py?")
        return
    
    # Convert list of dicts to HuggingFace Dataset
    dataset = Dataset.from_list(raw_data)
    # Split into training and evaluation sets
    dataset = dataset.train_test_split(test_size=args.test_size, seed=42)

    # 2. Load Tokenizer and Model
    print(f"Loading model and tokenizer: {args.model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForTokenClassification.from_pretrained(
            args.model_name, 
            num_labels=len(LABELS), 
            id2label=ID2LABEL, 
            label2id=LABEL2ID
        )
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load model/tokenizer. Check dependencies! Error: {e}")
        return

    # Initialize Data Collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # 3. Preprocess Data
    print("Tokenizing and aligning labels...")
    tokenized_datasets = dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer), 
        batched=True, 
        remove_columns=dataset["train"].column_names # Remove raw tokens/tags columns
    )

    # 4. Training Arguments
    training_args = TrainingArguments(
        # FIX: Access the positional argument using args.output_directory
        output_dir=args.output_directory,
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch", # Evaluate performance after each epoch
        save_strategy="epoch",
        load_best_model_at_end=True, # Load the model with the best evaluation score
        logging_dir='../dataset/ner_animal_model/logs',
        logging_steps=50,
    )
    
    # 5. Trainer Initialization and Training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Ensure output directory exists before training starts
    os.makedirs(args.output_directory, exist_ok=True)
    
    print("\n" + "="*50)
    print("Starting NER fine-tuning... (This may take some time)")
    print("="*50 + "\n")
    
    trainer.train()
    
    # 6. Save Final Model
    # FIX: Access the positional argument using args.output_directory
    trainer.save_model(args.output_directory) 
    tokenizer.save_pretrained(args.output_directory) 
    print(f"Training complete. Best model and tokenizer saved to {args.output_directory}")

if __name__ == "__main__":
    main()