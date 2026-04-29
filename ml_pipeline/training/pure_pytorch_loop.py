import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import List

class ExamTopicDataset(Dataset):
    """Custom PyTorch Dataset for loading Exam text and mapping it to correct topics."""
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
    output_dir: str = "../../saved_models/pytorch_topic_predictor",
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5
):
    """
    Explicit PyTorch training loop for BERT fine-tuning.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)

    # 2. Split and Tokenize
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
    
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

    # 3. Create Datasets and DataLoaders
    train_dataset = ExamTopicDataset(train_encodings, train_labels)
    val_dataset = ExamTopicDataset(val_encodings, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 4. Set up Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 5. Training Loop
    best_val_loss = float('inf')

    for epoch in range(epochs):
        print(f"\n======== Epoch {epoch + 1} / {epochs} ========")
        
        # --- TRAINING PHASE ---
        model.train()
        total_train_loss = 0
        
        train_progress = tqdm(train_loader, desc="Training", leave=False)
        for batch in train_progress:
            # Move batch tensors to device (GPU/CPU)
            optimizer.zero_grad() # Clear previous gradients
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=target_labels)
            loss = outputs.loss
            total_train_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            train_progress.set_postfix({'Loss': f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- VALIDATION PHASE ---
        model.eval()
        total_val_loss = 0
        correct_predictions = 0
        
        with torch.no_grad(): # Disable gradient calculation for validation speed
            val_progress = tqdm(val_loader, desc="Validating", leave=False)
            for batch in val_progress:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                target_labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask, labels=target_labels)
                loss = outputs.loss
                total_val_loss += loss.item()
                
                # Calculate accuracy
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                correct_predictions += (predictions == target_labels).sum().item()

        avg_val_loss = total_val_loss / len(val_loader)
        accuracy = correct_predictions / len(val_dataset)
        
        print(f"Training Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f} | Validation Accuracy: {accuracy:.4f}")
        
        # 6. Save BEST model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f">>> Validation loss improved! Saving model to {output_dir}")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    print("\nTraining Complete!")
    return model, tokenizer

if __name__ == "__main__":
    mock_questions = [
        "Explain backpropagation in deep neural networks.",
        "How does a transformer handle self attention?",
        "Define gradient descent and learning rate.",
        "What is the time complexity of bubble sort?",
        "Explain Dijkstra's algorithm for shortest paths."
    ]
    mock_labels = [0, 0, 0, 1, 1] 

    # train_pure_pytorch(mock_questions, mock_labels, num_labels=2)
