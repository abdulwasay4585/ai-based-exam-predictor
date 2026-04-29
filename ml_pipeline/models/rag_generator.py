import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
import sys
import os

# Add data_processing to path to import FaissExamIndexer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_processing.faiss_indexer import FaissExamIndexer

class ExamRAGPipeline:
    """
    Retrieval-Augmented Generation (RAG) pipeline for Exam Generation.
    1. Retrieves historically relevant exam questions from the FAISS database for a given topic.
    2. Feeds the retrieved past questions into a generative language model (LLM) as context.
    3. Prompts the LLM to generate a brand new, highly probable exam question based on that context.
    """
    def __init__(
        self, 
        indexer: FaissExamIndexer, 
        generator_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    ):
        self.indexer = indexer
        # Forcing CPU: PyTorch modern binaries do not support old GPU architectures
        self.device = torch.device("cpu")
        
        print(f"Loading Generation Model '{generator_model_name}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model_name, trust_remote_code=True)
        
        # Upgrading engine to CausalLM (Next Token Prediction) required for advanced Instruct models
        # CRITICAL: Slicing the float mapping natively from Float32 down to Float16 bypasses OS SWAP Memory freezing!
        self.model = AutoModelForCausalLM.from_pretrained(
            generator_model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(self.device)

    def generate_creative_topic_title(self, topic_keywords_str: str) -> str:
        """
        Synthesizes a formalized and creative course topic title given a set of NLP keywords.
        Uses Chat Templates to explicitly instruct the 0.5B AI format.
        """
        messages = [
            {"role": "system", "content": "You are a university academic coordinator. Convert the provided keywords into a short, formal academic course title. Only output the title string itself with zero extra text."},
            {"role": "user", "content": f"Keywords: {topic_keywords_str}"}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=15, 
            do_sample=False,         
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        output_ids = outputs[0][inputs.input_ids.shape[1]:]
        title = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        
        if "\n" in title: title = title.split("\n")[0].strip()
        if ":" in title: title = title.split(":")[-1].strip()
        if "\"" in title: title = title.replace("\"", "").strip()
            
        if len(title) < 5 or title.lower() in topic_keywords_str.lower():
            return "Advanced Course Analytics"
        return title.title()

    def generate_predicted_question(
        self, 
        topic_query: str, 
        context_count: int = 4, # Reduced context slightly for Causal LM speed
        max_new_tokens: int = 400
    ) -> str:
        # Step 1: Retrieval Phase
        print(f"\n[RAG Retrieval] Searching historical database for: '{topic_query}'")
        retrieved_results = self.indexer.search(topic_query, top_k=context_count)
        
        if not retrieved_results:
            return "No historical context found for this topic to anchor the generation."
            
        context_texts = [text for score, text in retrieved_results]
        aggregated_context = " | ".join(context_texts)
        
        print(f"[RAG Context Formed] -> Found {len(context_texts)} previous related vectors.")

        # Step 2: Prompt Engineering Phase using System Directives
        messages = [
            {"role": "system", "content": "You are a PhD Physics Professor. Create an extraordinarily challenging mathematical university exam question. CRITICAL INSTRUCTION: You MUST wrap EVERY single mathematical variable, equation, or fraction natively inside LaTeX dollar signs (e.g. $F=ma$, $\\frac{1}{2}$). Do NEVER use parenthesis for equations. ONLY output the raw question itself."},
            {"role": "user", "content": f"Context Facts:\n{aggregated_context}\n\nTask: Draft a highly complex mathematical physics question calculating '{topic_query}'."}
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Step 3: Generation Phase
        print("[RAG Generation] Synthesizing new exam question...")
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=True,          
            temperature=0.85,        # HIGH variance forces every question in the 5-loop to be totally mathematically distinct
            top_p=0.95,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Strip prompt matrix off the causal response payload natively
        output_ids = outputs[0][inputs.input_ids.shape[1]:]
        generated_question = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        
        # Post-process: Automatically crush annoying LLM conversational filler
        if "Certainly" in generated_question or "Here is" in generated_question:
            if "Question:" in generated_question:
                generated_question = generated_question.split("Question:", 1)[-1].strip()
            elif "\n\n" in generated_question:
                generated_question = generated_question.split("\n\n", 1)[-1].strip()
                
        return generated_question

    def generate_model_answer(
        self, 
        exam_question: str, 
        course_notes_context: str,
        max_new_tokens: int = 800
    ) -> str:
        """
        Uses the LLM to generate a comprehensive, high-scoring model answer 
        for a predicted exam question based on provided scraped course notes.
        """
        print(f"\n[RAG Answer Generation] Synthesizing model answer for: '{exam_question}'")
        
        # Protect against HuggingFace context token overflow by clamping max context to ~600 words
        safe_context = course_notes_context[:3000]
        
        prompt = (
            f"Based strictly on this course material:\n"
            f"\"{safe_context}\"\n\n"
            f"Provide a comprehensive, highly detailed, step-by-step academic answer to the following Exam Question. Write an extensive response with multiple sentences and deep explanations covering all implications.\n"
            f"Exam Question: {exam_question}\n"
            f"Extensive Step-by-Step Answer:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        
        model_answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        # Clean up any trailing split sentences
        if model_answer and model_answer[-1] not in ('.', '!', '?'):
            model_answer = model_answer.rsplit('.', 1)[0] + "."
            
        return model_answer

# ------------ Example Usage ------------
if __name__ == "__main__":
    historical_exam_bank = [
        "Explain backpropagation and calculating gradients.",
        "How do self-attention mechanisms in transformers calculate weights?",
        "Describe the time complexity for merge sort algorithms.",
        "How does a convolutional neural network extract local features using filters?",
        "Explain the vanishing gradient problem in deep Recurrent Neural Networks."
    ]
    
    indexer = FaissExamIndexer()
    indexer.build_index(historical_exam_bank)
    
    rag_pipeline = ExamRAGPipeline(indexer=indexer)
    target_topic = "Neural Network Gradients"
    
    predicted_question = rag_pipeline.generate_predicted_question(
        topic_query=target_topic, 
        context_count=2
    )
    
    print("\n================ RAG OUTPUT ================")
    print(f"Predicted Question:\n>>> {predicted_question}")
    print("============================================")
