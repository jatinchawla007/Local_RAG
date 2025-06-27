# 📚 Local RAG on *The Art of Electronics* (1200+ Pages)

This is a **fully local Retrieval-Augmented Generation (RAG)** system built to enable intelligent question-answering on *The Art of Electronics* — a foundational text in electronics.  
It runs **without any RAG framework** and is designed to work with any local LLM, currently tested with **Gemma-3B (Google)** through **LM Studio**.

---

## ✨ Features

- 🔍 Local semantic search over 1200+ pages of *The Art of Electronics*
- 🧠 Embeddings generated using **MPNet-base-v2** from HuggingFace
- 🤖 LLM-agnostic design — plug in any local model (Gemma, Mistral, etc.)
- 💬 Answer questions from the book with high relevance via vector retrieval
- 🔒 Fully offline & private setup — no external API required

---



## 🧪 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/art-of-electronics-rag.git
cd art-of-electronics-rag
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` typically includes:

### 3. Generate or Load Embeddings

> Skip this if you already have `embeddings/` generated.

This script:
- Splits the book into text chunks
- Generates embeddings using `sentence-transformers/all-mpnet-base-v2`
- Stores them in a CSV, can use open source libraries like FAISS

### 4. Run the Query Pipeline



Sample flow:
- User asks: *"Explain the difference between BJTs and MOSFETs."*
- Retriever fetches top-k relevant chunks
- Prompt is created and sent to Gemma-3B via LM Studio
- Answer is returned based on book content

---

## 🤖 Using Gemma-3B with LM Studio

Ensure LM Studio is:
- Running locally (default: `http://localhost:1234/v1`)
- Hosting **Gemma-3B** with chat interface enabled


Example payload:
```json
{
  "model": "gemma-3b",
  "messages": [
    {"role": "system", "content": "You are a helpful electronics expert."},
    {"role": "user", "content": "Explain voltage dividers."}
  ]
}
```

---

## 🛠️ Customization

- Replace `Gemma` with any other model (Mistral, LLama, Phi, etc.)
- Tune chunk size and retrieval `top_k` values in `retriever.py`
- Add citation or source highlighting for better transparency

---

## 📘 Credits

- *The Art of Electronics* — Horowitz & Hill  
- [MPNet](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) for embeddings  
- [LM Studio](https://lmstudio.ai) for local LLM integration  
- FAISS for efficient vector search

---

## 🛡️ Disclaimer

This is an educational project.  
Please ensure you own a legitimate copy of *The Art of Electronics* before using this tool.  
Not intended for redistribution of copyrighted content.
