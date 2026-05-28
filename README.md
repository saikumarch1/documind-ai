# DocuMind AI

**Ask anything about any PDF. Get precise answers with exact page citations.**

Powered by local LLMs — zero API cost, runs entirely on your laptop.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Llama 3.2 via Ollama |
| Embeddings | all-MiniLM-L6-v2 (HuggingFace) |
| Vector Store | FAISS (Facebook AI) |
| Orchestration | LangChain 1.3 |
| PDF Parsing | PyPDF |
| UI | Streamlit |
| Language | Python 3.13 |

---

## Architecture

```
PDF Upload
    │
    ▼
┌─────────────────────────┐
│     PyPDFLoader          │  Load raw text from every page
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  RecursiveCharacter      │  Split into 1000-char chunks
│  TextSplitter            │  with 200-char overlap
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  all-MiniLM-L6-v2        │  Convert each chunk into a
│  Embedding Model         │  384-dimensional vector
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  FAISS Vector Store      │  Store and index all vectors
│  (saved to disk)         │  for millisecond search
└────────────┬────────────┘
             │
    ┌────────┘
    │   User asks a question
    ▼
┌─────────────────────────┐
│  Embed Question          │  Same MiniLM model converts
│  (same MiniLM model)     │  question into a vector
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Cosine Similarity       │  Find top-3 most relevant
│  Search (FAISS)          │  chunks from the index
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Prompt Template         │  Combine: system instruction
│  (LangChain)             │  + retrieved chunks + question
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Llama 3.2               │  Generate grounded answer
│  (local via Ollama)      │  with page number citations
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Streamlit Chat UI       │  Display answer + source tags
│  (Light / Dark mode)     │  + relevance scores
└─────────────────────────┘
```

---

## Key Features

- Upload **any PDF** — research papers, textbooks, contracts, lecture notes
- Every answer includes **exact page numbers**
- **Relevance scores** show how confident the retrieval was
- **Light and Dark mode** toggle
- Llama 3.2 runs **locally via Ollama** — no OpenAI, no billing
- FAISS index **saved to disk** — no re-indexing on restart
- Adjustable **top-k retrieval** slider (2–6 sources)

---

## Installation

**Prerequisites:** Python 3.10+ and [Ollama](https://ollama.com/download) installed.

```bash
# 1. Clone the repo
git clone https://github.com/saikumarch1/documind-ai.git
cd documind-ai

# 2. Create virtual environment
python -m venv venv --without-pip
venv\Scripts\python.exe -m ensurepip --upgrade
venv\Scripts\activate

# 3. Install dependencies
venv\Scripts\python.exe -m pip install -r requirements.txt

# 4. Pull the LLM
ollama pull llama3.2

# 5. Run the app
venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## How RAG Works

**RAG = Retrieval-Augmented Generation.** It solves two LLM problems: models cannot read your private files, and they hallucinate facts.

1. **Ingest** — PDF split into 1000-char chunks, 200-char overlap prevents context loss at boundaries
2. **Embed** — Each chunk converted to a 384-dim vector using MiniLM
3. **Index** — Vectors stored in FAISS for fast nearest-neighbour lookup
4. **Retrieve** — Question embedded, cosine similarity finds top-3 chunks
5. **Generate** — Chunks + question sent to Llama 3.2; answer grounded in document only

The model cannot hallucinate — it only uses what was retrieved.

---

## Project Structure

```
documind-ai/
├── app.py                 # Main Streamlit application
├── build_index.py         # Pre-build FAISS index from PDF
├── rag_chain.py           # Terminal RAG chain (dev/testing)
├── load_pdf.py            # PDF loading utility
├── test_ollama.py         # LLM connection test
├── test_retrieval.py      # Retrieval quality test
├── faiss_index/
│   ├── index.faiss        # Vector index
│   └── index.pkl          # Chunk metadata
├── requirements.txt
└── README.md
```

---

## Architectural Decisions

| Decision | Reason |
|----------|--------|
| FAISS over ChromaDB | Faster on CPU, no server needed, saves to disk |
| MiniLM over OpenAI embeddings | Free, local, 384-dim sufficient for RAG |
| Llama 3.2 over GPT-4 | Zero cost, works offline, fits any laptop |
| chunk_overlap=200 | Prevents answers being cut off at chunk edges |
| k=3 retrieval | Balances context richness vs prompt length |

---

## Evaluation

| Metric | Value |
|--------|-------|
| Embedding model size | 90 MB |
| Embedding dimensions | 384 |
| Chunk size | 1000 chars |
| Chunk overlap | 200 chars |
| Retrieval method | Cosine similarity |
| LLM size | 2 GB (Llama 3.2 3B) |
| Avg response time | 15–45 sec (CPU) |

---

## Future Improvements

- [ ] HNSW indexing for faster retrieval at scale
- [ ] Multi-PDF support with document selector
- [ ] Conversation memory across sessions
- [ ] RAGAS evaluation metrics
- [ ] Docker containerization
- [ ] Hugging Face Spaces deployment

---

## Author

**Cheruku Sai Kumar** — B.Tech CSE, Lovely Professional University

[LinkedIn](http://www.linkedin.com/in/cheruku-sai-kumar) · [GitHub](https://github.com/saikumarch1)

---

## License

MIT — free to use, modify, and distribute.
