import torch
import pandas as pd
import json
import os
from typing import List, Dict
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

def evaluate_topic_classifier(
    model_path: str,
    test_data_path: str = "ml_pipeline/data/topic_data_test.json",
    mapping_path: str = "saved_models/topic_classifier/topic_mapping.json"
):
    print(f"--- Evaluating Topic Classifier: {model_path} ---")
    if not os.path.exists(test_data_path):
        print(f"Error: Test data not found at {test_data_path}")
        return

    # Load mapping
    with open(mapping_path, 'r') as f:
        id_to_topic = json.load(f)
    topic_to_id = {v: int(k) for k, v in id_to_topic.items()}

    # Load data
    df = pd.read_json(test_data_path, lines=True)
    texts = df['text'].tolist()
    true_topics = df['topic'].tolist()
    true_labels = [topic_to_id[t] for t in true_topics]

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    predictions = []
    print(f"Processing {len(texts)} samples...")
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            outputs = model(**inputs)
            pred_id = torch.argmax(outputs.logits, dim=-1).item()
            predictions.append(pred_id)

    acc = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, target_names=list(id_to_topic.values()))
    
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", report)
    
    return {"accuracy": acc, "report": report}

def evaluate_rag_generation(
    model_id: str = "saved_models/qwen_ft",
    test_data_path: str = "ml_pipeline/data/sciq_test.json",
    num_samples: int = 20
):
    """
    Evaluates the generation quality. 
    Note: Real evaluation would use ROUGE/BLEU. 
    This is a placeholder for structural validation.
    """
    print(f"--- Evaluating RAG Generator: {model_id} ---")
    if not os.path.exists(test_data_path):
        print(f"Error: Test data not found at {test_data_path}")
        return

    df = pd.read_json(test_data_path, lines=True).sample(num_samples)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()

    results = []
    for _, row in df.iterrows():
        context = row['support']
        ground_truth = row['question']
        
        prompt = f"Context: {context}\n\nQuestion: Based on the context, "
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the generated part
            prediction = generated_text[len(prompt):].strip()
        
        results.append({
            "context": context[:50] + "...",
            "ground_truth": ground_truth,
            "prediction": prediction
        })

    print(f"Evaluated {num_samples} samples. Sample Output:")
    print(json.dumps(results[0], indent=2))
    
    return results

if __name__ == "__main__":
    # Example usage:
    # evaluate_topic_classifier("saved_models/topic_classifier")
    # evaluate_rag_generation()
    pass
