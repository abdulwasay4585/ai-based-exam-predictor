# AI Exam Predictor: The Academic Oracle

[![Deployment](https://img.shields.io/badge/Deployment-Hugging%20Face-blue?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/abdulwasay4585/ai-exam-predictor)
[![Technical Report](https://img.shields.io/badge/Report-IEEE%20Format-red?style=for-the-badge&logo=latex)](./report_latex/report.pdf)
[![Tech Stack](https://img.shields.io/badge/Stack-FastAPI%20%7C%20React%20%7C%20PyTorch-green?style=for-the-badge)](https://github.com/abdulwasay4585/ai-based-exam-predictor)

**AI Exam Predictor** is a state-of-the-art full-stack platform designed to revolutionize academic preparation. By integrating **Retrieval-Augmented Generation (RAG)**, **Transformer-based Classification**, and **Probabilistic Trend Analysis**, the system acts as an "Academic Oracle" that identifies high-probability exam topics and synthesizes tailored study material directly from course lecture notes.

---

## Project Motivation

In the modern academic landscape, students are overwhelmed by vast quantities of digital resources. Traditional study methods involve manual review of thousands of pages of lecture notes, which is often subjective and inefficient. This project aims to:
1.  **Reduce Cognitive Load**: Automate the identification of "Hot Topics" and core concepts.
2.  **Ensure Contextual Accuracy**: Use RAG to ground AI generation in the user's specific course material, preventing hallucinations.
3.  **Provide Probabilistic Insights**: Use historical data and momentum-based heuristics to predict what will actually appear in the next exam.

---

## Key Features

- **Intelligent Topic Analysis**: Uses **BERT (SciQ-Net)** to identify the academic domain and **KeyBERT** for high-fidelity keyword extraction using Maximal Marginal Relevance (MMR).
- **Probabilistic Trend Forecasting**: Implements a custom heuristic engine using **Exponential Moving Average (EMA)** and **Linear Trend Slopes** to predict topic likelihood.
- **RAG-Powered Question Generation**: Utilizes **Qwen-2.5** and **T5** architectures grounded by **FAISS** vector retrieval to generate contextually accurate questions.
- **Automated Solution Synthesis**: Provides detailed, step-by-step model answers with full support for **LaTeX** mathematical rendering via KaTeX.
- **Premium UI/UX**: A responsive React interface featuring **Framer Motion** animations, **Tailwind CSS** glassmorphic styling, and real-time "Thinking" status indicators.
- **Production Ready**: Fully containerized with **Docker** and deployed via a robust **CI/CD pipeline** (GitHub Actions).

---

## Technical Stack

### **Frontend (Presentation Layer)**
- **Framework**: React.js 18 (Functional Components, Hooks)
- **State Management**: Local Storage persistence for session recovery.
- **Styling**: Tailwind CSS (Custom Academic Palette)
- **Animations**: Framer Motion (Staggered entries, Spring transitions)
- **Icons**: Lucide-React
- **Rendering**: KaTeX (Math) & ReactMarkdown (Rich Text)

### **Backend (Orchestration Layer)**
- **API Framework**: FastAPI (High-performance asynchronous endpoints)
- **Server**: Uvicorn
- **Orchestration**: Python 3.10
- **Document Processing**: PyMuPDF (fitz) for high-speed PDF stream handling.

### **Intelligence Layer (ML Pipeline)**
- **Core Framework**: PyTorch
- **Transformers**: HuggingFace (BERT-base, T5-small, Qwen-2.5-0.5B-Instruct)
- **Vector Database**: FAISS (IndexFlatIP for cosine similarity)
- **NLP Engine**: SpaCy (en_core_web_sm), Scikit-Learn (TF-IDF)
- **Embeddings**: all-MiniLM-L6-v2 (384-dimensional dense vectors)

---

## Project Structure

```text
├── backend/                # FastAPI application
│   ├── app/                # API routes (main.py, schemas.py)
│   └── uploads/            # Temporary buffer for PDF processing
├── frontend/               # React.js application
│   ├── src/                # App.js (Main Logic), index.css (Design System)
│   └── public/             # Static icons and assets
├── ml_pipeline/            # Core AI Logic
│   ├── models/             # Transformer wrappers (rag_generator.py, classifier.py)
│   ├── data_processing/    # faiss_indexer.py, pdf_extractor.py
│   ├── training/           # fine-tuning scripts (bert_finetuner.py)
│   └── utils/              # nlp_engine.py, trend_predictor.py
├── report_latex/           # IEEE Technical Report (4 pages)
├── Dockerfile              # Multi-stage production build (React Build -> Python Run)
└── requirements.txt        # Full dependency list
```

---

## Workflow

1.  **Ingestion**: User uploads a PDF. `PyMuPDF` extracts the text, which is then cleaned via regex to remove noise (URLs, headers, footers).
2.  **Indexing**: The text is chunked recursively and embedded using `all-MiniLM-L6-v2`. Vectors are stored in a local `FAISS` index.
3.  **Classification**: `SciQ-Net` (BERT) identifies the subject matter to refine the generation prompt's "Academic Persona".
4.  **Trend Analysis**: The `ExamTrendPredictor` calculates a momentum score based on keyword frequencies and temporal slopes.
5.  **Retrieval**: When a question is generated, the system retrieves the top-K relevant chunks from the FAISS index.
6.  **Generation**: `Qwen-2.5` or `T5` synthesizes a question and answer based on the retrieved context, formatted in LaTeX.

---

## Installation & Setup

### **Prerequisites**
- Python 3.9+
- Node.js & npm
- Docker (Optional)

### **Local Development**

1. **Clone the Repository**
   ```bash
   git clone https://github.com/abdulwasay4585/ai-based-exam-predictor.git
   cd ai-based-exam-predictor
   ```

2. **Backend Setup**
   ```bash
   cd backend
   pip install -r ../requirements.txt
   python -m spacy download en_core_web_sm
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

3. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   npm run build # For production or npm start for dev
   ```

---

## Model Benchmarks

- **SciQ-Net (BERT Classifier)**:
  - Accuracy: **92.4%**
  - Loss: **0.10 (Test)**
  - Epochs: **10 (Fine-tuned)**
- **RAG Generation**:
  - Context Retrieval Latency: **<50ms**
  - Hallucination Rate: **Significantly reduced** compared to zero-shot LLMs.

---
*Created for AI-341: Deep Neural Network Project, GIK Institute. 2026*
