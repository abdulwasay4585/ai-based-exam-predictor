import sys
import os
import tempfile
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

# Append the ML pipeline to the system path so FastAPI can access our ML algorithms
current_dir = os.path.dirname(os.path.abspath(__file__))
ml_pipeline_dir = os.path.abspath(os.path.join(current_dir, '../../ml_pipeline'))
sys.path.append(ml_pipeline_dir)

# Import the core orchestrator we just built
from main_pipeline import MasterExamOrchestrator
from data_processing.pdf_extractor import extract_text_from_pdf, clean_exam_text

app = FastAPI(
    title="AI Exam Predictor API",
    description="Endpoint interface mapping frontends to our backend LLM/RAG generation systems.",
    version="1.1.0"
)

# Crucial for allowing React (localhost:3000) to hit FastAPI (localhost:8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the heavy ML models globally on startup exactly once
print("Pre-loading ML Orchestrator (FAISS / HuggingFace)...")
try:
    orchestrator = MasterExamOrchestrator(current_year=2023)
except Exception as e:
    print(f"Failed to load ML models: {e}")
    orchestrator = None

class PredictionRequest(BaseModel):
    notes: List[str]
    topic_historical_frequencies: Dict[str, Dict[int, int]]

@app.get("/")
def health_check():
    return {"status": "ok", "message": "FastAPI ML Predictive Backend is online."}

@app.post("/api/predict-exam")
async def predict_exam_from_notes(request: PredictionRequest):
    """
    Accepts raw structural syllabus notes and a dictionary of historical frequencies.
    Returns internally generated predicted topics, questions, and model answers via JSON.
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="ML Models are unavailable on the server.")

    try:
        if not request.notes or not request.topic_historical_frequencies:
            raise HTTPException(status_code=400, detail="Notes and historical frequencies are missing.")

        # Execute the AI generation pipeline synchronously
        result = orchestrator.generate_full_exam_material(
            topic_histories=request.topic_historical_frequencies,
            course_notes_corpus=request.notes
        )
        
        return {
            "success": True,
            "data": result
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc() # Print to UVICORN console for debugging
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-notes")
async def upload_pdf_notes(files: List[UploadFile] = File(...)):
    """
    Endpoint for uploading up to 10 raw PDF files (max 50MB total) to be dynamically processed into real semantic notes.
    """
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum of 10 PDF files are allowed.")
    
    total_size = 0
    all_chunks = []
    
    for file in files:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail=f"Only .pdf files are supported. Found: {file.filename}")
            
        file_content = await file.read()
        total_size += len(file_content)
        
        if total_size > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Total file size exceeds the 50MB limit.")
            
        temp_fd, temp_path = tempfile.mkstemp(suffix=".pdf")
        os.close(temp_fd)
        
        with open(temp_path, "wb") as buffer:
            buffer.write(file_content)
            
        try:
            raw_text = extract_text_from_pdf(temp_path)
            cleaned_text = clean_exam_text(raw_text)
            chunks = [chunk.strip() for chunk in cleaned_text.split('\n\n') if len(chunk.strip()) > 50]
            all_chunks.extend(chunks)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    if not all_chunks:
        all_chunks = ["The NLP Extractor failed to find readable text vectors in these PDF files."]
        
    return {
        "filename": f"{len(files)} files uploaded",
        "extracted_notes": all_chunks,
        "message": "Files parsed successfully."
    }
