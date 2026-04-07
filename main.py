from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
from app.query import query as rag_query
from app.ingest import ingest
from app.tracer import log_score

app = FastAPI(
    title="TechPrep AI API",
    description="Production RAG Pipeline for Technical Interview Prep",
    version="2.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    user_id: str = "default"


class FeedbackRequest(BaseModel):
    trace_id: str
    helpful: bool


@app.get("/")
def root():
    return {"message": "TechPrep AI is running!"}


@app.post("/query")
def ask_question(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )
    result = rag_query(request.question, request.user_id)
    return result


@app.post("/feedback")
def submit_feedback(request: FeedbackRequest):
    score = 1.0 if request.helpful else 0.0
    comment = "helpful" if request.helpful else "not helpful"
    log_score(request.trace_id, score, comment)
    return {"message": "Feedback recorded"}


@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    allowed = [".txt", ".md", ".pdf"]  # Added PDF
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail="Only .txt, .md and .pdf files allowed"
        )

    save_path = f"./data/raw/{file.filename}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"message": f"{file.filename} uploaded successfully!"}


@app.post("/ingest")
def ingest_documents():
    try:
        ingest()
        return {"message": "Documents ingested successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/eval")
def run_evaluation():
    from app.eval import run_eval
    results = run_eval()
    return {"results": results}