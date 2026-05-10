import torch
import pandas as pd
import argparse
import os
import json
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from sklearn.model_selection import train_test_split
from typing import List

class ExamTopicDataset(Dataset):
    def __init__(self, encodings: dict, labels: List[int]):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train_bert_for_topic_prediction(
    train_path: str = "ml_pipeline/data/topic_data_train.json",
    test_path: str = "ml_pipeline/data/topic_data_test.json",
    model_name: str = "bert-base-uncased",
    output_dir: str = "saved_models/topic_classifier",
    epochs: int = 3,
    batch_size: int = 16,
    num_samples: int = None
):
    print(f"Loading data from {train_path}...")
    if not os.path.exists(train_path):
        print(f"Error: Data file not found at {train_path}.")
        return

    train_df = pd.read_json(train_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)

    if num_samples:
        train_df = train_df.sample(min(num_samples, len(train_df)))
        test_df = test_df.sample(min(num_samples // 5, len(test_df)))

    unique_topics = train_df['topic'].unique().tolist()
    topic_to_id = {topic: i for i, topic in enumerate(unique_topics)}
    id_to_topic = {i: topic for topic, i in topic_to_id.items()}
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "topic_mapping.json"), 'w') as f:
        json.dump(id_to_topic, f)

    train_texts = train_df['text'].tolist()
    train_labels = train_df['topic'].map(topic_to_id).tolist()
    val_texts = test_df['text'].tolist()
    val_labels = test_df['topic'].map(topic_to_id).tolist()

    print(f"Detected {len(unique_topics)} topics. Initializing model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(unique_topics))

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

    train_dataset = ExamTopicDataset(train_encodings, train_labels)
    val_dataset = ExamTopicDataset(val_encodings, val_labels)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=False,
        use_cpu=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # Force CPU for demo
    print("Beginning fine-tuning on CPU...")
    trainer.train()

    print(f"Saving finalized model and tokenizer to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BERT for Topic Prediction.")
    parser.add_argument("--train_path", type=str, default="ml_pipeline/data/topic_data_train.json")
    parser.add_argument("--test_path", type=str, default="ml_pipeline/data/topic_data_test.json")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="saved_models/topic_classifier")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to use (None for full).")
    
    args = parser.parse_args()
    
    train_bert_for_topic_prediction(
        train_path=args.train_path,
        test_path=args.test_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_samples=args.num_samples
    )
