import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict

class QuestionEvaluator:
    """
    Evaluation suite to systematically compare AI-generated predicted exam questions 
    against the actual, ground-truth questions that ultimately appeared on the real exam.
    """
    def __init__(self, semantic_model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading Semantic Evaluator Model: {semantic_model_name}...")
        # We use SentenceTransformers because it is state-of-the-art for semantic 
        # textual similarity (STS) comparisons between two sentences.
        self.semantic_model = SentenceTransformer(semantic_model_name)

    def calculate_cosine_similarity(self, predicted_text: str, actual_text: str) -> float:
        """
        Uses dense vector embeddings to check if the generated question asks 
        the exact same conceptual thing as the real question, even if phrased completely differently.
        """
        pred_emb = self.semantic_model.encode(predicted_text, convert_to_tensor=True)
        act_emb = self.semantic_model.encode(actual_text, convert_to_tensor=True)
        
        cosine_score = util.pytorch_cos_sim(pred_emb, act_emb)
        return cosine_score.item()

    def evaluate_test_set(
        self, 
        predictions: List[str], 
        ground_truths: List[str]
    ) -> Dict[str, float]:
        """
        Runs a comprehensive batch evaluation over a set of predicted vs actual questions.
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and Ground Truths lists must be universally identical in length.")

        semantic_scores = []
        jaccard_scores = []
        
        print(f"Evaluating {len(predictions)} predicted constraint pairs against reality...")
        for pred, actual in zip(predictions, ground_truths):
            # 1. Semantic Score (Did it capture the underlying 'idea' of the exam question?)
            sem_score = self.calculate_cosine_similarity(pred, actual)
            semantic_scores.append(sem_score)

            # 2. Jaccard Lexical Overlap Score (Did it use the same exact academic buzzwords?)
            pred_set = set(pred.lower().split())
            act_set = set(actual.lower().split())
            intersection = len(pred_set.intersection(act_set))
            union = len(pred_set.union(act_set))
            jaccard_scores.append(intersection / union if union != 0 else 0)

        return {
            "Average_Semantic_Similarity": np.mean(semantic_scores),
            "Average_Lexical_Overlap": np.mean(jaccard_scores),
            "Perfect_Semantic_Matches (>85%)": len([s for s in semantic_scores if s >= 0.85]),
            "Total_Evaluated": len(predictions)
        }

if __name__ == "__main__":
    # ================= EVALUATION STRATEGY & TEST CASES =================
    
    # 1. Ground Truths: The exact questions from the 2024 final paper (Revealed post-exam)
    reality_exam_paper = [
        "Explain the mathematical mechanism causing vanishing gradients in Recurrent Neural Networks.",
        "What is the average case time complexity of QuickSort?",
        "Define Artificial Intelligence."
    ]
    
    # 2. Predicted Questions: What our AI RAG pipeline generated the day before the exam
    ai_predicted_questions = [
        # TEST CASE A: High Semantic Match, Low Lexical (Good AI Prediction)
        # It predicted the exact concept, but phrased it differently.
        "How do standard RNNs suffer from vanishing gradients during backpropagation through time?",
        
        # TEST CASE B: Complete Miss (AI Failure)
        # AI incorrectly thought Dijkstra graphs would be on the exam instead of sorting.
        "Describe Dijkstra's shortest path algorithm.",
        
        # TEST CASE C: Word-for-Word Accuracy (Perfect Prediction)
        "Define Artificial Intelligence."
    ]

    evaluator = QuestionEvaluator()
    results = evaluator.evaluate_test_set(ai_predicted_questions, reality_exam_paper)
    
    print("\n--- AI Model vs Reality Final Evaluation Report ---")
    for metric, score in results.items():
        if "Average" in metric:
            print(f"{metric}: {score:.2%}")
        else:
            print(f"{metric}: {score}")
