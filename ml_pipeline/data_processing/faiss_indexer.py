import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple

class FaissExamIndexer:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initializes the embedding model and FAISS vector database.
        Using a dedicated Sentence Transformer is highly recommended for FAISS vector 
        retrieval (over standard BERT) because it natively fine-tunes for Cosine similarity.
        """
        # Forcing CPU: PyTorch modern binaries do not support old GPU architectures (like MX130 / sm_50)
        self.device = torch.device("cpu")
        print(f"Loading embedding model '{model_name}' on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        self.dimension = None
        self.index = None
        
        # Metadata storage (maps FAISS integer IDs back to raw question text)
        self.metadata = []

    def _mean_pooling(self, model_output, attention_mask):
        """Standard mean pooling to compress sentence tokens into a single standard feature vector."""
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Converts raw text into dense vector embeddings."""
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            
        # Perform pooling
        embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Normalize embeddings for Cosine Similarity (standard for FAISS Inner Product metric)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # FAISS strictly requires float32 NumPy arrays
        return embeddings.cpu().numpy().astype('float32')

    def build_index(self, texts: List[str]):
        """Generates embeddings and builds the core FAISS index."""
        print(f"Generating embeddings for {len(texts)} exam questions...")
        embeddings = self.generate_embeddings(texts)
        
        self.dimension = embeddings.shape[1]
        
        # IndexFlatIP uses Inner Product (which equals Cosine Similarity if vectors are normalized)
        # Note: For Euclidean Distance (L2), use faiss.IndexFlatL2(self.dimension)
        self.index = faiss.IndexFlatIP(self.dimension)
        
        print("Adding vectors to FAISS index...")
        self.index.add(embeddings)
        # BUG FIX: Completely overwrite previous session metadata instead of appending old ghost inputs via .extend()
        self.metadata = list(texts)
        
        print(f"Index built! Total items in FAISS: {self.index.ntotal}")

    def search(self, query: str, top_k: int = 3) -> List[Tuple[float, str]]:
        """
        Queries the FAISS index for the most semantically similar exam questions.
        """
        if not self.index:
            raise ValueError("FAISS index is empty. Call build_index() first.")
            
        query_embedding = self.generate_embeddings([query])
        
        # D = Distances (scores), I = Indices of the matches in the FAISS db
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:  # -1 means FAISS didn't find enough matches in the whole db
                results.append((dist, self.metadata[idx]))
                
        return results

    def save_index(self, filepath: str = "../../data/processed/exam_faiss.index"):
        """Saves the FAISS index to disk."""
        if self.index:
            faiss.write_index(self.index, filepath)
            print(f"FAISS index saved to {filepath}")


# ------------ Example Usage ------------
if __name__ == "__main__":
    exam_bank = [
        "What are the implications of the P vs NP problem in computer science?",
        "Explain backpropagation and calculating gradients.",
        "How do self-attention mechanisms in transformers calculate weights?",
        "Describe the time complexity for merge sort algorithms.",
        "How does a convolutional neural network extract local features using filters?"
    ]
    
    # indexer = FaissExamIndexer()
    # indexer.build_index(exam_bank)
    
    # Arbitrary semantically loosely-linked query
    # search_query = "Tell me about NLP and attention layers in models"
    # print(f"\nSearching FAISS for: '{search_query}'")
    
    # matches = indexer.search(search_query, top_k=2)
    # for score, text in matches:
    #     print(f"[Score: {score:.3f}] {text}")
    pass
