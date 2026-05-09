import sys
import os
import fitz
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
from data_processing.pdf_extractor import clean_exam_text

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
    topic_historical_frequencies: Dict[str, Dict[int, int]] = {}
    model_choice: str = "option1"

class AnswerRequest(BaseModel):
    question: str
    context: str

class ExportRequest(BaseModel):
    payload: Dict[str, Any]

class SolveRequest(BaseModel):
    notes: List[str]
    model_choice: str = "option1"

@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "FastAPI ML Predictive Backend is online."}

@app.post("/api/predict-exam")
async def predict_exam_from_notes(request: PredictionRequest):
    if not orchestrator:
        raise HTTPException(status_code=503, detail="ML Models are unavailable.")
    try:
        result = orchestrator.generate_full_exam_material(
            topic_histories=request.topic_historical_frequencies,
            course_notes_corpus=request.notes,
            model_choice=request.model_choice
        )
        return {"success": True, "data": result}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-notes")
async def upload_pdf_notes(files: List[UploadFile] = File(...)):
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum of 10 PDF files are allowed.")
    total_size = 0
    all_chunks = []
    for file in files:
        file_content = await file.read()
        total_size += len(file_content)
        if total_size > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Total size exceeds 50MB.")
        
        try:
            # Optimization: Open PDF from memory stream to avoid disk I/O
            doc = fitz.open(stream=file_content, filetype="pdf")
            raw_text = []
            for page in doc:
                raw_text.append(page.get_text())
            doc.close()
            
            cleaned_text = clean_exam_text("\n".join(raw_text))
            # Intelligent chunking by paragraph
            chunks = [chunk.strip() for chunk in cleaned_text.split('\n\n') if len(chunk.strip()) > 50]
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            
    if not all_chunks:
        all_chunks = ["No text found."]
    return {"extracted_notes": all_chunks}

@app.post("/api/generate-answer")
async def generate_answer(request: AnswerRequest):
    if not orchestrator:
        raise HTTPException(status_code=503, detail="ML Models are unavailable.")
    try:
        answer = orchestrator.generate_answer_for_question(request.question, request.context)
        return {"success": True, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/export-exam")
async def export_exam(request: ExportRequest):
    data = request.payload
    lines = [f"# Exam: {data.get('predicted_topic')}", f"Confidence: {data.get('overall_confidence')}", "\n===\n"]
    for q in data.get('questions', []):
        lines.append(f"### Q{q['id']} ({q['probability']})")
        lines.append(q['question'])
        if 'answer' in q:
            lines.append(f"\nAnswer: {q['answer']}")
        lines.append("\n---\n")
    return {"success": True, "content": "\n".join(lines)}

@app.post("/api/solve-exam")
async def solve_exam(request: SolveRequest):
    if not orchestrator:
        raise HTTPException(status_code=503, detail="ML Models unavailable.")
    try:
        result = orchestrator.solve_uploaded_questions(request.notes, request.model_choice)
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/classify-exam")
async def classify_exam(request: SolveRequest):
    if not orchestrator:
        raise HTTPException(status_code=503, detail="ML Models unavailable.")
    try:
        result = orchestrator.classify_uploaded_text(request.notes, request.model_choice)
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

    @app.exception_handler(404)
    async def custom_404_handler(request, __):
        return FileResponse(os.path.join(static_dir, "index.html"))
else:
    print(f"Warning: Static directory {static_dir} not found. Running API only mode.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
