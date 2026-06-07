# DocuMind 🧠📄
**A production-quality RAG system for PDF question-answering — built free, runs locally.**

Ask any question about any PDF and get accurate, grounded answers with source citations.

---

## Stack
| Component | Technology | Why |
|---|---|---|
| PDF parsing | PyMuPDF | Fast, accurate, handles complex layouts |
| Embeddings | `all-MiniLM-L6-v2` | Free, 384-dim, runs locally after first download |
| Vector store | FAISS | Millisecond similarity search, no server needed |
| LLM | `Mistral-7B-Instruct-v0.3` | Free via HF Inference API, strong instruction following |
| Orchestration | LangChain | RetrievalQA chain wires retriever → LLM cleanly |
| UI | Gradio | Real app feel, shareable link option |

---

## Quick Start

```bash
# 1. Clone & set up environment
git clone <your-repo>
cd documind
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure your HuggingFace token
cp .env.example .env
# Edit .env and paste your token from huggingface.co → Settings → Access Tokens

# 3. Verify everything works
python verify_setup.py

# 4. Launch the app
python app.py
```

---

## Project Structure
```
documind/
├── src/
│   ├── config.py          # All tuneable settings
│   ├── ingestion.py       # PDF loading + chunking
│   ├── embeddings.py      # Embedding + FAISS index management
│   ├── retriever.py       # Similarity search logic
│   ├── chain.py           # LangChain RetrievalQA + prompt
│   └── evaluate.py        # Retrieval quality evaluation
├── data/                  # Uploaded PDFs (gitignored)
├── vectorstore/           # FAISS index (gitignored)
├── logs/                  # Query logs for evaluation
├── app.py                 # Gradio UI entry point
├── verify_setup.py        # Environment health check
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Evaluation Results
*(Fill in after completing Step 7)*

| Metric | Value |
|---|---|
| Retrieval latency (avg) | ~XXX ms |
| Precision@4 on 10-query test set | X/10 |
| Hallucination rate: RAG vs direct LLM | ~X% vs ~X% |

---

## Key Design Decisions

**Chunk size 500 / overlap 50** — Balances context completeness against embedding noise. Each chunk ≈ 120 tokens, well within MiniLM's 256-token window.

**TOP_K = 4** — Feeds the LLM enough context for multi-part questions without overwhelming its context window or increasing hallucination risk.

**Temperature 0.1** — Keeps the LLM faithful to retrieved content rather than improvising.
