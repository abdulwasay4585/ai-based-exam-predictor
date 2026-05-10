import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Any, Optional

class ExamRAGPipeline:
    def __init__(
        self, 
        indexer: Optional[Any] = None, 
        generator_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    ):
        self.indexer = indexer
        self.device = torch.device("cpu")
        print(f"Loading RAG Generation Model '{generator_model_name}' on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            generator_model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        self.model.eval()
        print("RAG Model loaded successfully on CPU.")

    def generate_creative_topic_title(self, topic_keywords_str: str) -> str:
        messages = [
            {"role": "system", "content": "You are a university academic coordinator. Convert the provided keywords into a short, formal academic course title. Only output the title string itself."},
            {"role": "user", "content": f"Keywords: {topic_keywords_str}"}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([prompt], return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        output_ids = outputs[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    def generate_predicted_question(
        self, 
        topic_query: str, 
        context_override: Optional[str] = None,
        max_new_tokens: int = 150 
    ) -> str:
        print(f"\n[RAG Generation] Synthesizing question for: '{topic_query}'")
        aggregated_context = (context_override[:500] if context_override else "General academic concepts.")
        messages = [
            {"role": "system", "content": "You are an exam Professor. Create a challenging exam question using LaTeX ($$..$$) for math. Be concise and specific."},
            {"role": "user", "content": f"Context: {aggregated_context}\n\nGenerate one exam question about '{topic_query}':"}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([prompt], return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.8, top_p=0.9, pad_token_id=self.tokenizer.eos_token_id)
        output_ids = outputs[0][inputs.input_ids.shape[1]:]
        result = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        print(f"[RAG Generation] Question generated ({len(result)} chars)")
        return result

    def generate_model_answer(
        self, 
        exam_question: str, 
        course_notes_context: str,
        max_new_tokens: int = 1500 
    ) -> str:
        print(f"\n[RAG Answer] Generating answer for: '{exam_question[:60]}...'")
        best_context = course_notes_context[:4500] 
        messages = [
            {"role": "system", "content": "You are a PhD Professor. Answer the provided exam question in a detailed, comprehensive manner. REQUIREMENT: Use step-by-step logical reasoning. Format all mathematical variables and formulas in LaTeX using DOUBLE dollar signs (e.g. $$x^2$$). Do not be brief; provide a thorough academic explanation based on the course material. If the material is insufficient, use your expertise to fill the gaps professionally."},
            {"role": "user", "content": f"COURSE MATERIAL:\n{best_context}\n\nQUESTION: {exam_question}\n\nProvide a complete, professional, and detailed answer:"}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([prompt], return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.3, top_p=0.9, repetition_penalty=1.1, pad_token_id=self.tokenizer.eos_token_id)
        output_ids = outputs[0][inputs.input_ids.shape[1]:]
        result = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        print(f"[RAG Answer] Answer generated ({len(result)} chars)")
        return result
