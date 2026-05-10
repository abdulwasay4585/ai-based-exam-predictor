import json
from typing import Dict, List, Any
import os
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration

# Import our custom ML modules
from models.trend_predictor import ExamTrendPredictor
from models.rag_generator import ExamRAGPipeline
from utils.nlp_engine import ExamNLPEngine

class MasterExamOrchestrator:
    def __init__(self, current_year: int = 2023):
        print("Initializing Master ML Orchestrator...")
        self.device = torch.device("cpu")
        print(f"Compute device: {self.device}")

        self.predictor = ExamTrendPredictor(current_year=current_year)
        self.nlp_engine = ExamNLPEngine()
        self.rag = ExamRAGPipeline(indexer=None) 
        
        self.custom_model_path = "saved_models/custom_sciq_model"
        self.custom_gen_path = "saved_models/custom_sciq_gen"
        self.custom_model = None
        self.custom_tokenizer = None
        self.custom_gen_model = None
        self.custom_gen_tokenizer = None

        # Load SciQ Classifier (Option 2)
        if os.path.exists(self.custom_model_path):
            print(f"Loading SciQ-Net Classifier on CPU...")
            try:
                self.custom_tokenizer = AutoTokenizer.from_pretrained(self.custom_model_path)
                self.custom_model = AutoModelForSequenceClassification.from_pretrained(self.custom_model_path)
                self.custom_model.eval()
                print("SciQ Classifier loaded.")
            except Exception as e:
                print(f"Warning: Could not load classifier: {e}")

        # Load T5 Generator (Option 2)
        if os.path.exists(self.custom_gen_path):
            print(f"Loading SciQ-Net T5 Generator on CPU...")
            try:
                self.custom_gen_tokenizer = T5Tokenizer.from_pretrained(self.custom_gen_path)
                self.custom_gen_model = T5ForConditionalGeneration.from_pretrained(self.custom_gen_path)
                self.custom_gen_model.eval()
                print("SciQ T5 Generator loaded.")
            except Exception as e:
                print(f"Warning: Could not load generator: {e}")

    def reset_session(self):
        pass

    def generate_full_exam_material(
        self, 
        topic_histories: Dict[str, Dict[int, int]], 
        course_notes_corpus: List[str],
        model_choice: str = "option1"
    ) -> Dict[str, Any]:
        print(f"\n--- Starting Generation Pipeline Mode: {model_choice} ---")
        full_text = " ".join(course_notes_corpus)
        
        # 1. NLP Topic Analysis
        hot_topics, distributions = self.nlp_engine.extract_hot_topics(full_text)
        frequent_concepts, concept_weights = self.nlp_engine.identify_frequent_concepts(full_text)
        
        primary_topic = hot_topics[0] if hot_topics else "General Concept"
        
        # 2. Dynamic confidence calculation
        dynamic_conf = self.nlp_engine.calculate_dynamic_confidence(hot_topics, distributions, full_text)
        if model_choice == "option2" and self.custom_model:
            inputs = self.custom_tokenizer(full_text[:512], return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.custom_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                conf, pred_id = torch.max(probs, dim=-1)
                dynamic_conf = (dynamic_conf + conf.item()) / 2  # Blend NLP + classifier

        # 3. Question Generation
        generated_questions = []
        snippets = []
        for concept in frequent_concepts[:5]:
            matches = re.finditer(re.escape(concept), full_text, re.IGNORECASE)
            for m in list(matches)[:1]:
                start = max(0, m.start() - 200)
                end = min(len(full_text), m.end() + 300)
                snippets.append(full_text[start:end])

        corpus_preprocessed = self.nlp_engine._preprocess(full_text[:2000])

        num_questions = 3
        for i in range(num_questions):
            print(f"\n[Pipeline] Generating question {i+1}/{num_questions}...")
            context_chunk = snippets[i % len(snippets)] if snippets else full_text[:1000]
            
            if model_choice == "option2" and self.custom_gen_model:
                input_text = f"generate question: {context_chunk[:500]}"
                inputs = self.custom_gen_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = self.custom_gen_model.generate(**inputs, max_length=100)
                    question_text = self.custom_gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                question_text = self.rag.generate_predicted_question(
                    topic_query=primary_topic, 
                    context_override=context_chunk,
                    max_new_tokens=128
                )
            
            q_prob = self.nlp_engine.calculate_question_probability(question_text, hot_topics, corpus_preprocessed=corpus_preprocessed)
            generated_questions.append({
                "id": i + 1,
                "question": question_text,
                "probability": f"{q_prob}%",
                "prediction_tag": primary_topic
            })

        print("\n*** Full Pipeline Execution Complete ***\n")
        return {
            "predicted_topic": primary_topic,
            "hot_topics": hot_topics,
            "topic_distribution": distributions,
            "frequent_concepts": frequent_concepts,
            "overall_confidence": f"{dynamic_conf * 100:.1f}%",
            "questions": generated_questions,
            "retrieved_context_used": full_text[:2000],
            "engine": "Qwen-2.5 (Instruction Mode)" if model_choice == "option1" else "SciQ-Net (BERT+T5 Engine)"
        }

    def solve_uploaded_questions(self, uploaded_text: List[str], model_choice: str = "option1") -> Dict[str, Any]:
        print(f"\n--- Starting SOLVE Mode ---")
        full_text = " ".join(uploaded_text)
        raw_questions = re.split(r'(?:\d+\.|\?|\n)(?=\s*[A-Z])', full_text)
        
        cleaned_questions = []
        for q in raw_questions:
            q = q.strip()
            if len(q) < 40 or len(q) > 600: continue
            if "@" in q or "http" in q: continue
            if "University" in q or "Dept" in q or "Department" in q: continue
            if not any(word in q.lower() for word in ["what", "how", "explain", "describe", "define", "calculate", "derive", "why", "list"]):
                if not q.endswith("?"): continue
            cleaned_questions.append(q if q.endswith("?") else q + "?")
        
        target_questions = cleaned_questions[:5] if cleaned_questions else [full_text[:300]]
        solved_questions = []
        
        for i, q in enumerate(target_questions):
            print(f"\n[Solve] Answering question {i+1}/{len(target_questions)}...")
            if model_choice == "option2" and self.custom_gen_model:
                input_text = f"answer question: {q}"
                inputs = self.custom_gen_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = self.custom_gen_model.generate(**inputs, max_length=200)
                    answer = self.custom_gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                answer = self.rag.generate_model_answer(exam_question=q, course_notes_context=full_text)
            
            solved_questions.append({
                "id": i + 1,
                "question": q,
                "answer": answer,
                "probability": "100% (Input Source)"
            })
            
        return {
            "predicted_topic": "Direct Paper Solution",
            "hot_topics": ["Exam Solving"],
            "topic_distribution": {"Solving": 100},
            "overall_confidence": "Direct Analysis",
            "questions": solved_questions,
            "retrieved_context_used": full_text[:1000],
            "engine": "RAG-Enhanced Solver"
        }

    def classify_uploaded_text(self, texts: List[str], model_choice: str = "option1") -> Dict[str, Any]:
        full_text = " ".join(texts)
        hot_topics, distributions = self.nlp_engine.extract_hot_topics(full_text)
        dynamic_conf = self.nlp_engine.calculate_dynamic_confidence(hot_topics, distributions, full_text)
        
        # Use Qwen RAG for Option 1 topic identification
        if model_choice == "option1":
            primary_label = self.rag.generate_creative_topic_title(
                ", ".join(hot_topics[:3])
            ) if hot_topics and hot_topics[0] != "Insufficient Content" else "General Academic Subject"
        else:
            # Use SciQ classifier for Option 2
            primary_label = "Academic Subject"
            if self.custom_model:
                inputs = self.custom_tokenizer(full_text[:512], return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    outputs = self.custom_model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    conf, pred_id = torch.max(probs, dim=-1)
                    dynamic_conf = (dynamic_conf + conf.item()) / 2
                    mapping_path = os.path.join(self.custom_model_path, "topic_mapping.json")
                    if os.path.exists(mapping_path):
                        with open(mapping_path, 'r') as f:
                            mapping = json.load(f)
                        primary_label = mapping.get(str(pred_id.item()), "Science Concepts")
            else:
                primary_label = hot_topics[0] if hot_topics else "Unclassified"

        return {
            "predicted_topic": primary_label,
            "hot_topics": hot_topics,
            "topic_distribution": distributions,
            "overall_confidence": f"{dynamic_conf * 100:.1f}%",
            "questions": [],
            "engine": "Qwen-2.5 (Topic Analyzer)" if model_choice == "option1" else "SciQ-Net + LDA Analyzer"
        }

    def generate_answer_for_question(self, question: str, context: str) -> str:
        return self.rag.generate_model_answer(exam_question=question, course_notes_context=context)
