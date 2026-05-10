import torch
import pandas as pd
import os
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from tqdm import tqdm

class SciQGenDataset(Dataset):
    def __init__(self, tokenizer, data, max_len=256):
        self.tokenizer = tokenizer
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        input_text = f"generate question: {row['support']}"
        target_text = row['question']

        input_enc = self.tokenizer(
            input_text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt"
        )
        target_enc = self.tokenizer(
            target_text, max_length=64, padding='max_length', truncation=True, return_tensors="pt"
        )

        return {
            "input_ids": input_enc["input_ids"].squeeze(),
            "attention_mask": input_enc["attention_mask"].squeeze(),
            "labels": target_enc["input_ids"].squeeze()
        }

def train_custom_generator(
    data_path="ml_pipeline/data/sciq_train.json",
    model_name="t5-small",
    output_dir="saved_models/custom_sciq_gen",
    epochs=5,
    batch_size=8,
    num_samples=500
):
    print(f"--- Training Custom Generator (Option 2) on {num_samples} samples ---")
    if not os.path.exists(data_path):
        print("Data not found.")
        return

    df = pd.read_json(data_path, lines=True).sample(num_samples)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    device = torch.device("cpu")
    model.to(device)

    dataset = SciQGenDataset(tokenizer, df)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    model.train()
    for epoch in range(epochs):
        loop = tqdm(loader, desc=f"Gen Epoch {epoch+1}/{epochs}")
        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

    print(f"Saving custom generator to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    train_custom_generator()
