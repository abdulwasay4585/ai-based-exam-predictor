import torch
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
    """
    Custom PyTorch Dataset for loading Exam text and mapping it to correct topics.
    """
    def __init__(self, encodings: dict, labels: List[int]):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Convert dictionary of lists to a dictionary of tensors for PyTorch
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train_bert_for_topic_prediction(
    texts: List[str], 
    labels: List[int], 
    num_labels: int,
    model_name: str = "bert-base-uncased",
    output_dir: str = "../../saved_models/bert_topic_predictor",
    epochs: int = 3,
    batch_size: int = 8
):
    """
    Fine-tunes a pretrained BERT model on exam text to predict associated topic categories.
    
    Args:
        texts (List[str]): Raw or cleaned exam question texts.
        labels (List[int]): Integer-encoded labels representing the correct topics.
        num_labels (int): The total number of distinct topics.
        model_name (str): HuggingFace pretrained model string.
    """
    
    print(f"Loading '{model_name}' tokenizer & model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # 1. Train / Validation Split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # 2. Tokenize Data
    print("Tokenizing datasets...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

    # 3. Create PyTorch Datasets
    train_dataset = ExamTopicDataset(train_encodings, train_labels)
    val_dataset = ExamTopicDataset(val_encodings, val_labels)

    # 4. Set up Training Arguments for HuggingFace Trainer
    training_args = TrainingArguments(
        output_dir=output_dir,             # Path to save trained weights
        num_train_epochs=epochs,           # Total epochs
        per_device_train_batch_size=batch_size,  
        per_device_eval_batch_size=batch_size * 2,
        warmup_steps=100,                  # Linear warmup 
        weight_decay=0.01,                 # L2 regularization to prevent overfitting
        logging_dir='./logs',              
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,       # Ensures the best overall model is kept
        fp16=torch.cuda.is_available()     # Mixed precision training if GPU is present
    )

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,                         
        args=training_args,                  
        train_dataset=train_dataset,         
        eval_dataset=val_dataset             
    )

    # 6. Run Training Loop
    print(f"Beginning fine-tuning on device: {trainer.args.device}")
    trainer.train()

    # 7. Save Final Model and Tokenizer
    print(f"Saving finalized model and tokenizer to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer

# Example usage interface
if __name__ == "__main__":
    # Simulate data inputs that would normally load from our NLP Preprocessor Phase
    mock_questions = [
        "Explain backpropagation in deep neural networks.",
        "How does a transformer handle self attention?",
        "Define gradient descent and learning rate.",
        "What is the time complexity of bubble sort?",
        "Explain Dijkstra's algorithm for shortest paths."
    ]
    # Simulate categorical integer labels 
    # e.g., 0: Deep Learning, 1: Algorithms
    mock_labels = [0, 0, 0, 1, 1] 

    # In a real environment, you run this block.
    # train_bert_for_topic_prediction(mock_questions, mock_labels, num_labels=2)
    pass
