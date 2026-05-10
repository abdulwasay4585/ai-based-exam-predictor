import torch
import json
import pandas as pd
import os
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import List
from sklearn.metrics import accuracy_score

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

def train_pure_pytorch(
    texts: List[str], 
    labels: List[int], 
    num_labels: int,
    model_name: str = "bert-base-uncased",
    output_dir: str = "saved_models/custom_sciq_model",
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5
):
    device = torch.device("cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)

    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1, random_state=42)
    
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

    train_dataset = ExamTopicDataset(train_encodings, train_labels)
    val_dataset = ExamTopicDataset(val_encodings, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_accuracy = 0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for batch in train_progress:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=target_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_progress.set_postfix({'loss': f"{loss.item():.4f}"})

        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                target_labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(target_labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1} - Validation Accuracy: {acc:.4f}")

        if acc > best_accuracy:
            best_accuracy = acc
            print(f"New Best Accuracy! Saving to {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    print(f"Training Complete. Best Accuracy: {best_accuracy:.4f}")
    return model, tokenizer

def run_custom_training(num_samples=1000):
    train_path = "ml_pipeline/data/topic_data_train.json"
    
    if not os.path.exists(train_path):
        print("Data not found.")
        return

    train_df = pd.read_json(train_path, lines=True).sample(num_samples)
    
    unique_topics = train_df['topic'].unique().tolist()
    topic_to_id = {topic: i for i, topic in enumerate(unique_topics)}
    id_to_topic = {i: topic for topic, i in topic_to_id.items()}
    
    os.makedirs("saved_models/custom_sciq_model", exist_ok=True)
    with open("saved_models/custom_sciq_model/topic_mapping.json", 'w') as f:
        json.dump(id_to_topic, f)

    texts = train_df['text'].tolist()
    labels = train_df['topic'].map(topic_to_id).tolist()

    train_pure_pytorch(
        texts=texts,
        labels=labels,
        num_labels=len(unique_topics),
        epochs=5,
        batch_size=16
    )

if __name__ == "__main__":
    run_custom_training()
