import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, precision_recall_fscore_support
from typing import List, Dict

def evaluate_model(
    model_path: str, 
    test_texts: List[str], 
    true_labels: List[int],
    target_names: List[str] = None
) -> Dict[str, float]:
    """
    Evaluates a trained BERT Sequence Classification model on a holdout dataset 
    and calculates Precision, Recall, and F1-Score.
    
    Args:
        model_path (str): Directory containing the saved PyTorch model and tokenizers.
        test_texts (List[str]): Raw text inputs to evaluate.
        true_labels (List[int]): Ground truth integer labels for the test texts.
        target_names (List[str], optional): Human-readable names for the integer classes.
        
    Returns:
        Dict[str, float]: Dictionary mapping metric names to their weighted scores.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {model_path} onto {device}...")
    
    try:
        # Load Fine-Tuned Model & Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    except OSError:
        print(f"Error: Model not found at '{model_path}'. Ensure training is complete first.")
        return {}

    model.to(device)
    model.eval()

    predicted_labels = []

    print("Running inference on test dataset...")
    # Disable gradient tracking to dramatically increase evaluation speed and reduce memory
    with torch.no_grad():
        for text in test_texts:
            # Tokenize input dynamically
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            inputs = {key: val.to(device) for key, val in inputs.items()}
            
            # Predict
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Extract highest confidence class
            prediction = torch.argmax(logits, dim=-1).item()
            predicted_labels.append(prediction)

    # Calculate aggregated metrics using sklearn
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, 
        predicted_labels, 
        average='weighted', # 'weighted' handles class imbalances safely
        zero_division=0
    )
    
    print("\n--- Detailed Classification Report ---")
    print(classification_report(true_labels, predicted_labels, target_names=target_names, zero_division=0))

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

if __name__ == "__main__":
    # Example Simulation:
    
    # 1. Point to the directory where your PyTorch/Trainer loop saved the model weights
    # model_dir = "../../saved_models/pytorch_topic_predictor"
    
    # 2. Provide unseen holdout test data
    # unseen_questions = ["Describe backpropagation.", "What is O(N) complexity?"]
    # actual_labels = [0, 1]
    
    # 3. Trigger evaluation
    # metrics = evaluate_model(
    #    model_dir, 
    #    unseen_questions, 
    #    actual_labels, 
    #    target_names=["Deep Learning", "Algorithms"]
    # )
    
    # print(f"Final F1-Score: {metrics.get('f1_score', 0):.2f}")
    pass
