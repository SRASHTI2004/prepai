# TechPrep AI 🚀
> Your personal AI-powered technical interview preparation assistant

![TechPrep AI Demo](techprep%20ai/screenshots/02-output.png)

👉 [**Try the live app →**](https://prepai-h8ekghmvjxyfapdgbc9dth.streamlit.app/)

---

## What it does

Ask any System Design, DSA, or AI/ML interview question — get accurate, 
cited answers from a curated knowledge base. No hallucinations.

---

## Screenshots

| Input Screen | Answer Output | Sources |
|---|---|---|
| ![Input](techprep%20ai/screenshots/01-input.png)) | ![Output](techprep%20ai/screenshots/02-output.png) | ![Sources](techprep%20ai/screenshots/03-sources.png) |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| FastAPI | REST API backend |
| Qdrant | Vector database for semantic search |
| Cohere | Reranking for better retrieval |
| Groq (Llama 3) | LLM for answer generation |
| Langfuse | Observability and tracing |
| Streamlit | Frontend UI |
| BM25 + Dense | Hybrid retrieval strategy |

---

## System Performance

| Metric | Score |
|---|---|
| Answer Relevancy | 0.84 / 1.0 |
| Faithfulness | 0.24 / 1.0 |
| Avg Total Latency | ~6754ms |
| Retrieval Latency | ~5771ms |
| Generation Latency | ~983ms |

---

## Architecture

User Query
↓
Hybrid Search (BM25 + Dense Retrieval)
↓
Cohere Reranking
↓
Groq LLM (Answer Generation)
↓
Langfuse (Tracing + Observability)
↓
Final Answer with Sources

---

## Project Structure

techprep-ai/
├── app/
│   ├── retrieval.py    # Hybrid BM25 + dense search
│   ├── query.py        # RAG query pipeline
│   ├── ingest.py       # Document ingestion
│   ├── reranker.py     # Cohere reranking
│   └── eval.py         # LLM-as-judge evaluation
├── main.py             # FastAPI backend
├── ui.py               # Streamlit frontend
├── requirements.txt
└── .env.example

---

## Local Setup

### 1. Clone the repo
```bash
git clone https://github.com/SRASHTI2004/TechPrep-AI.git
cd TechPrep-AI
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add API keys
```bash
cp .env.example .env
```
Add your keys:

QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_key
GROQ_API_KEY=your_groq_key
COHERE_API_KEY=your_cohere_key
LANGFUSE_PUBLIC_KEY=your_langfuse_key
LANGFUSE_SECRET_KEY=your_langfuse_secret

### 5. Ingest documents
```bash
python -m app.ingest
```

### 6. Run the app
```bash
# Terminal 1
uvicorn main:app --reload

# Terminal 2
streamlit run ui.py
```
Open → http://localhost:8501

---

## Built by

**Srashti Choudhary** — IT Graduate @ MAIT, Delhi  
Aspiring AI/Backend Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/srashti-choudhary)
[![GitHub](https://img.shields.io/badge/GitHub-SRASHTI2004-black)](https://github.com/SRASHTI2004)
