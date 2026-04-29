import spacy

# Load the English NLP model from spaCy
# You may need to install it first if you haven't: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def preprocess_text(text: str, return_as_string: bool = True):
    """
    Preprocesses text data using spaCy by performing:
    1. Tokenization
    2. Lowercasing
    3. Stopword removal
    4. Punctuation removal
    5. Lemmatization
    
    Args:
        text (str): The input string to process.
        return_as_string (bool): If True, returns a joined string of lemmas.
                                 If False, returns a list of lemmas.
                                 
    Returns:
        str or list: The processed text.
    """
    if not text or not isinstance(text, str):
        return "" if return_as_string else []
        
    # Process the text using the spaCy pipeline
    # We disable NER and Parser because we only need tokenization, pos, and lemmatization,
    # which makes the processing much faster.
    doc = nlp(text, disable=["ner", "parser"])
    
    processed_tokens = []
    
    for token in doc:
        # Check if token is a stop word or punctuation
        if not token.is_stop and not token.is_punct and not token.is_space:
            # Lemmatize and lowercase the token
            lemma = token.lemma_.lower()
            
            # Additional custom filtering can go here (e.g. ignoring numbers)
            # if lemma.isalpha(): 
            processed_tokens.append(lemma)
            
    if return_as_string:
        return " ".join(processed_tokens)
    
    return processed_tokens

# Example Usage
if __name__ == "__main__":
    sample_text = "The students are studying really hard for their final Artificial Intelligence exams!"
    
    print("Original Text:", sample_text)
    print("Processed Text (String):", preprocess_text(sample_text))
    print("Processed Text (List):", preprocess_text(sample_text, return_as_string=False))
