import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="TechPrep AI",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 TechPrep AI — Technical Interview Assistant")
st.caption("Your personal AI powered interview prep knowledge base")

# Sidebar
with st.sidebar:
    st.header("⚙️ Setup")

    uploaded_file = st.file_uploader(
        "Upload document (.txt, .md, .pdf)",
        type=["txt", "md", "pdf"]
    )

    if uploaded_file:
        if st.button("Ingest Document"):
            with st.spinner("Uploading..."):
                files = {"file": (
                    uploaded_file.name,
                    uploaded_file.getvalue()
                )}
                res = requests.post(f"{API_URL}/upload", files=files)
                if res.status_code == 200:
                    st.success("Uploaded!")
                    requests.post(f"{API_URL}/ingest")
                    st.success("Ingested!")
                else:
                    st.error("Upload failed")

    st.divider()
    st.header("📊 System Metrics")

    # Show metrics if available
    if "metrics" in st.session_state:
        m = st.session_state["metrics"]
        st.metric(
            "Total Latency",
            f"{m['total_latency_ms']}ms"
        )
        st.metric(
            "Retrieval Latency",
            f"{m['retrieval_latency_ms']}ms"
        )
        st.metric(
            "Generation Latency",
            f"{m['generation_latency_ms']}ms"
        )
        if "sources" in m and m["sources"]:
            st.caption("Sources used:")
            for s in m["sources"]:
                st.caption(f"• {s}")

# Main area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("💬 Ask a Question")

    category = st.selectbox(
        "Category",
        [
            "System Design",
            "DSA & Algorithms",
            "AI & ML Concepts",
            "About Me",
            "Custom Question"
        ]
    )

    sample_questions = {
        "System Design": [
            "What is load balancing?",
            "Explain CAP theorem",
            "What is a message queue?",
            "How does consistent hashing work?",
            "What is database sharding?",
        ],
        "DSA & Algorithms": [
            "What is dynamic programming?",
            "Explain BFS vs DFS",
            "What is a binary search tree?",
            "When to use a heap?",
            "What is time complexity of quicksort?",
        ],
        "AI & ML Concepts": [
            "What is RAG?",
            "What is a vector database?",
            "What are embeddings?",
            "What is fine tuning vs RAG?",
            "What is hallucination in LLMs?",
        ],
        "About Me": [
            "Tell me about your projects",
            "What are your technical skills?",
            "What are your career goals?",
            "Why should we hire you?",
            "What is your greatest strength?",
        ],
        "Custom Question": []
    }

    if category != "Custom Question":
        selected_q = st.selectbox(
            "Select question",
            sample_questions[category]
        )
    else:
        selected_q = ""

    custom_q = st.text_input("Or type your own question")
    final_question = custom_q if custom_q else selected_q

    if st.button("Get Answer 🔍", type="primary"):
        if final_question:
            with st.spinner("Searching knowledge base..."):
                res = requests.post(
                    f"{API_URL}/query",
                    json={
                        "question": final_question,
                        "user_id": "streamlit_user"
                    }
                )
                if res.status_code == 200:
                    data = res.json()
                    st.session_state["last_answer"] = data["answer"]
                    st.session_state["last_question"] = final_question
                    st.session_state["trace_id"] = data.get("trace_id")
                    st.session_state["metrics"] = {
                        "total_latency_ms": data["total_latency_ms"],
                        "retrieval_latency_ms": data["retrieval_latency_ms"],
                        "generation_latency_ms": data["generation_latency_ms"],
                        "sources": data.get("sources", [])
                    }
                    st.rerun()
                else:
                    st.error("Something went wrong")

with col2:
    st.subheader("🤖 AI Answer")

    if "last_answer" in st.session_state:
        st.info(f"**Q:** {st.session_state['last_question']}")
        st.write(st.session_state["last_answer"])

        st.divider()

        # Sources
        if st.session_state.get("metrics", {}).get("sources"):
            st.caption(
                f"📚 Sources: "
                f"{', '.join(st.session_state['metrics']['sources'])}"
            )

        # Feedback
        st.caption("Was this helpful?")
        c1, c2 = st.columns(2)

        with c1:
            if st.button("👍 Yes"):
                requests.post(
                    f"{API_URL}/feedback",
                    json={
                        "trace_id": st.session_state.get("trace_id"),
                        "helpful": True
                    }
                )
                st.success("Thanks!")

        with c2:
            if st.button("👎 No"):
                requests.post(
                    f"{API_URL}/feedback",
                    json={
                        "trace_id": st.session_state.get("trace_id"),
                        "helpful": False
                    }
                )
                st.warning("Feedback recorded")
    else:
        st.info("Ask a question to get started")

st.divider()

# Eval section
st.subheader("📊 Pipeline Evaluation")
col3, col4 = st.columns([1, 3])

with col3:
    if st.button("Run Ragas Eval"):
        with st.spinner("Running evaluation... (takes ~1 min)"):
            res = requests.get(f"{API_URL}/eval")
            if res.status_code == 200:
                st.session_state["eval_results"] = res.json()["results"]

with col4:
    if "eval_results" in st.session_state:
        r = st.session_state["eval_results"]
        e1, e2 = st.columns(2)
        e1.metric(
            "Faithfulness",
            f"{r['faithfulness']:.2f}"
        )
        e2.metric(
            "Answer Relevancy",
            f"{r['answer_relevancy']:.2f}"
        )