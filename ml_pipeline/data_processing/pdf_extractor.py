import re
import fitz  # PyMuPDF
import string

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts raw text from a PDF file using PyMuPDF.
    
    Args:
        pdf_path (str): Path to the PDF exam paper.
        
    Returns:
        str: Extracted raw text.
    """
    try:
        doc = fitz.open(pdf_path)
        extracted_text = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            # Extract plain text
            text = page.get_text()
            extracted_text.append(text)

        doc.close()
        return "\n".join(extracted_text)
    
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""


def clean_exam_text(text: str) -> str:
    """
    Cleans raw PDF text for downstream NLP processing (tokenization, embeddings).
    
    Args:
        text (str): Raw extracted text.
        
    Returns:
        str: Cleaned and normalized text.
    """
    if not text:
        return ""

    # 1. Standardize text encoding down to ASCII to remove weird hidden chars
    text = text.encode("ascii", "ignore").decode()
    
    # 2. Remove URLs (often found in references or footnotes)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

    
    # 3. Replace multiple spaces, tabs, and newlines with a single space or standard newline
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\t+', ' ', text)
    
    # We want to keep logical newlines (for question separation) but remove excessive ones
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 4. Remove unwanted bullet points/artifacts from PDF scraping (e.g., •, —, page numbers)
    # Using a regex to detect standalone numbers at the start or end of lines which are often page numbers.
    text = re.sub(r'(?m)^[0-9]{1,3}$', '', text) 
    text = text.replace('•', '')
    text = text.replace('—', '-')
    
    # 5. Optional lowercasing (Depends on whether your NLP model is cased or uncased)
    # text = text.lower()
    
    return text.strip()


# Example Usage
if __name__ == "__main__":
    # sample_pdf_path = "../../data/raw/2023_machine_learning_exam.pdf"
    # raw_text = extract_text_from_pdf(sample_pdf_path)
    # cleaned_results = clean_exam_text(raw_text)
    # print(cleaned_results[:500])
    pass
