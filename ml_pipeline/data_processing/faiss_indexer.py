import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple

class FaissExamIndexer:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.device = torch.device("cpu")
        print(f"Loading embedding model '{model_name}' on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        self.dimension = None
        self.index = None
        self.metadata = []

    def reset(self):
        """Resets the FAISS index and metadata for a fresh session."""
        print("Clearing FAISS index and metadata for new chat session...")
        self.index = None
        self.metadata = []
        self.dimension = None

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy().astype('float32')

    def build_index(self, texts: List[str]):
        print(f"Generating embeddings for {len(texts)} exam chunks...")
        embeddings = self.generate_embeddings(texts)
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        self.metadata = list(texts)
        print(f"Index built! Total items: {self.index.ntotal}")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, str]]:
        if not self.index:
            return []
        query_embedding = self.generate_embeddings([query])
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                results.append((dist, self.metadata[idx]))
        return results
