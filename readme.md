# 🚀 TechPrep AI — Technical Interview Knowledge Base

> A production-grade RAG system built for technical interview preparation.
> Query a curated knowledge base covering System Design, DSA, and AI/ML concepts using natural language.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red)
![Qdrant](https://img.shields.io/badge/Qdrant-Cloud-purple)

---

## 📌 What Is This?

As a final year CS student preparing for AI engineering roles, 
I needed a single tool that could answer technical interview 
questions across System Design, DSA, and AI/ML — grounded in 
real curated content, not hallucinated answers.

So I built TechPrep AI — a production-grade RAG pipeline that:
- Retrieves relevant content using **Hybrid Search**
- Reranks results using **Cohere Cross-Encoder**
- Generates answers using **Groq LLM**
- Tracks every query using **Langfuse Observability**
- Evaluates quality using a **Custom Eval Pipeline**

---

## 🏗️ Architecture

User Query
│
▼
┌─────────────────────────────────┐
│         Hybrid Retrieval        │
│  BM25 Search + Dense Search     │
│  Reciprocal Rank Fusion (RRF)   │
│  Cohere Cross-Encoder Reranking │
└─────────────────────────────────┘
│
▼
┌─────────────────────────────────┐
│         LLM Generation          │
│  Groq API (Llama 3.1 8B)        │
│  Context-grounded prompt        │
└─────────────────────────────────┘
│
▼
┌─────────────────────────────────┐
│         Observability           │
│  Langfuse Tracing               │
│  Latency + Cost Tracking        │
│  User Feedback Logging          │
└─────────────────────────────────┘

---

## 📊 Evaluation Results

| Metric | Score |
|--------|-------|
| Answer Relevancy | **0.84 / 1.0** |
| Faithfulness | **0.24 / 1.0** (improving with better data) |
| Avg Query Latency | **~6000ms** |
| Retrieval Latency | **~5500ms** |
| Generation Latency | **~800ms** |

> Evaluated on 5 golden questions using custom LLM-as-judge 
> eval pipeline powered by Groq.

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|------------|
| LLM | Groq API — Llama 3.1 8B / 70B |
| Vector DB | Qdrant Cloud |
| Embeddings | FastEmbed |
| Retrieval | BM25 + Dense + RRF |
| Reranking | Cohere Rerank v3 |
| Observability | Langfuse |
| Backend | FastAPI |
| Frontend | Streamlit |
| Language | Python 3.12 |

---

## 🔍 Key Features

### 1. Hybrid Search
Combines BM25 keyword search with dense vector search.
BM25 catches exact keyword matches while dense search 
captures semantic meaning. Results fused using 
Reciprocal Rank Fusion (RRF).

### 2. Cohere Cross-Encoder Reranking
After RRF fusion, top candidates are reranked using 
Cohere's cross-encoder model for higher precision retrieval.
This significantly improves answer quality over basic 
vector search alone.

### 3. Langfuse Observability
Every query is traced end-to-end with:
- Retrieval latency
- Generation latency  
- Total latency
- User feedback (helpful / not helpful)

### 4. Custom Eval Pipeline
LLM-as-judge evaluation pipeline using Groq that scores:
- Faithfulness — is the answer grounded in context?
- Answer Relevancy — does it answer the question?

### 5. PDF Support
Supports .txt, .md, and .pdf document ingestion — 
upload your own notes or textbooks.

---

## 📁 Project Structure

techprep-ai/
├── app/
│   ├── config.py       # All config and env vars
│   ├── loaders.py      # Document loaders (txt, md, pdf)
│   ├── chunkers.py     # Text chunking with metadata
│   ├── ingest.py       # Embedding and Qdrant upload
│   ├── retrieval.py    # Hybrid search + reranking
│   ├── query.py        # RAG pipeline + tracing
│   ├── eval.py         # Evaluation pipeline
│   ├── llm.py          # LLM clients
│   └── tracer.py       # Langfuse tracing
├── data/
│   └── raw/            # Knowledge base documents
├── main.py             # FastAPI backend
├── dashboard.py               # Streamlit frontend
├── requirements.txt
└── .env

---

## 🚀 How To Run

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/techprep-ai
cd techprep-ai
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\Activate.ps1  # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file:

GROQ_API_KEY=your_key
QDRANT_URL=your_url
QDRANT_API_KEY=your_key
COHERE_API_KEY=your_key
LANGFUSE_PUBLIC_KEY=your_key
LANGFUSE_SECRET_KEY=your_key

### 5. Add your documents
Put `.txt`, `.md`, or `.pdf` files in `data/raw/`

### 6. Ingest documents
```bash
python -m app.ingest
```

### 7. Run the app
```bash
# Terminal 1
uvicorn main:app --reload

# Terminal 2
streamlit run ui.py
```

---

## 💡 Why I Built This

As a final year CS student actively interviewing for 
AI engineering roles, I found no single tool that could 
answer technical questions across System Design, DSA, 
and AI/ML in a grounded, reliable way.

I built TechPrep AI to solve my own problem — and used 
it to prepare for interviews. The production-grade 
architecture (hybrid retrieval, reranking, observability, 
eval) reflects how real AI systems are built.

---

## 🔮 Future Improvements

- [ ] Add more high quality domain data
- [ ] Implement query caching for faster responses
- [ ] Add conversation memory for follow-up questions
- [ ] Deploy on cloud (Railway / Render)
- [ ] Add voice input support

---

## 👩‍💻 Author

**Srashti**
Final Year IT Student | Aspiring AI Engineer

- GitHub: https://github.com/SRASHTI2004
- LinkedIn: https://www.linkedin.com/in/srashti-choudhary
- Email: srashtichoudhary5@gmail.com