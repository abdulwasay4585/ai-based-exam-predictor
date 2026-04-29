from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import warnings

# Suppress PyTorch device warnings gracefully
warnings.filterwarnings("ignore")

# CRITICAL FIX: The NVIDIA MX130 throws "no kernel image" because sm_50 is deprecated.
# We physically map the transformer explicitly to the CPU to block the CUDA memory crash!
sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
bert_model = KeyBERT(model=sentence_model)

def extract_frequent_topics(
    documents: List[str], 
    num_topics: int = 5, 
    num_keywords_per_topic: int = 2
) -> Dict[str, List[str]]:
    """
    Extracts the most conceptually relevant academic phrases from an exam corpus 
    using deeply embedded Neural Attention (BERT) matrices.
    """
    if not documents or len(documents) == 0:
        return {}

    # Aggressive pre-filtering of the corpus strings to strip numbers and generic physics PDF noise
    import re
    cleaned_docs = []
    for doc in documents:
        clean_text = re.sub(r'\b(?:page|fig|figure|table|chapter|section|pdf)\b', '', doc.lower())
        clean_text = re.sub(r'\b\w{1,3}\b', '', clean_text) 
        clean_text = re.sub(r'[^a-zA-Z\s]', '', clean_text)  
        if len(clean_text.strip()) > 10:
            cleaned_docs.append(clean_text)

    # Fallback to pure document if regex brutally deleted everything
    if not cleaned_docs:
        cleaned_docs = documents

    # 1. Flatten the syllabus perfectly into a macroscopic academic sequence
    full_text = " ".join(cleaned_docs)
    
    # 2. Extract Keyphrases using Cosine Similarity natively matching BERT tokens!
    # Explicitly using Maximal Marginal Relevance (MMR) technique to rigorously force high diversity
    # and violently block duplicate synonym clusters (e.g. "wheel motion" vs "motion of the wheel")
    extracted_tags = bert_model.extract_keywords(
        full_text,
        keyphrase_ngram_range=(1, 3), # Expansion to 3-word tensors captures deep Physics syntax
        stop_words='english',
        use_mmr=True,
        diversity=0.7,
        top_n=num_topics * 2 # Fetch excess bounds strictly for filtering
    )
    
    # Structure of extracted_tags: [("quantum mechanics", 0.655), ("magnetic field", 0.612)]
    
    topics_keywords = {}
    valid_keyphrases = [kw[0] for kw in extracted_tags]
    
    # 3. Simulate original dictionary block logic mapping cleanly backwards to main_pipeline.py
    for i in range(min(num_topics, len(valid_keyphrases))):
        topics_keywords[f"Topic {i + 1}"] = [valid_keyphrases[i]]
        
    return topics_keywords

# Example Usage
if __name__ == "__main__":
    # In a real pipeline, these would be the strings output by your `nlp_preprocessor.py`
    sample_exam_questions = [
        "Explain the fundamental concepts of supervised machine learning and provide examples.",
        "How do artificial neural networks use backpropagation to minimize the loss function?",
        "Define natural language processing and how recurrent neural networks are applied to it.",
        "Compare linear regression with logistic regression in the context of classification.",
        "Describe the architecture of a transformer model and the self-attention mechanism.",
        "What is the difference between K-Means clustering and hierarchical clustering algorithms?",
        "Discuss the ethical implications of deploying biased artificial intelligence systems."
    ]
    
    # Note: min_df=2 might fail on this tiny 7-sentence dataset, so for the tiny example 
    # we would normally set min_df=1, but for real exam papers min_df=2 is better.
    # To make this example runnable, let's redefine the exact vectorizer locally:
    vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, stop_words='english')
    matrix = vectorizer.fit_transform(sample_exam_questions)
    
    lda = LatentDirichletAllocation(n_components=3, random_state=42)
    lda.fit(matrix)
    
    features = vectorizer.get_feature_names_out()
    print("--- Extracted Topics ---\n")
    for i, topic in enumerate(lda.components_):
        keywords = [features[idx] for idx in topic.argsort()[:-6:-1]]
        print(f"Topic {i+1}: {', '.join(keywords)}")
