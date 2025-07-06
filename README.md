# RAG Chatbot with LangChain and Gemma

This project is a Retrieval-Augmented Generation (RAG)-based application that allows users to upload various document formats (**PDF, TXT, DOCX, MD**) and ask natural language questions. The system retrieves relevant content from the documents and generates human-like answers using a **local LLM (Gemma)**.

---

## Features

- Upload support for `.pdf`, `.txt`, `.docx`, and `.md` files.
- Automatic text extraction, chunking, and vectorization of content.
- Local vector database (**ChromaDB**) for similarity-based retrieval.
- Embedding with **all-MiniLM-L6-v2** from HuggingFace.
- Answer generation using **Gemma 2B** model running locally via **Ollama**.
- Source document name is shown along with the generated answer.
- Predefined responses for greetings and casual interactions.
- Relevance threshold check to avoid hallucinations or unrelated answers.

---

## Tech Stack

| Component         | Description                                 |
|------------------|---------------------------------------------|
| LLM              | Gemma 2B via Ollama (Local)                  |
| Embedding Model  | all-MiniLM-L6-v2 via HuggingFace            |
| Vector Store     | Chroma (Local persistent vector database)    |
| Text Splitter    | RecursiveCharacterTextSplitter               |
| UI Framework     | Streamlit (for file upload and chat interface) |
| RAG Framework    | LangChain                                    |

---

## How It Works

### 1. Document Upload and Chunking

- User uploads a document via Streamlit.
- Document is parsed and split into overlapping text chunks (~800 characters with 200 overlap).

### 2. Embedding and Storage

- Each chunk is embedded using MiniLM and stored in ChromaDB.

### 3. Query Handling

- User submits a natural language query.
- The system checks for predefined friendly phrases (e.g., “hi”, “thanks”).
- If it's a valid query, the same embedding model vectorizes the question.

### 4. Retrieval and Answer Generation

- Top 3 most relevant chunks are retrieved based on cosine similarity.
- If similarity score ≥ 0.2, context is passed to the local Gemma model for answer generation.
- The system also displays the source file name(s) associated with the answer.

---

## Directory Structure

```
RAG_APP/
│
├── data/                      # Uploaded and processed documents
├── chroma/                    # Vector database storage
├── app.py                     # Streamlit frontend
├── rag_logic.py               # Core document handling and QA logic
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (if using external APIs)
```

---

## Installation

### 1. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Run Ollama and Pull Gemma

```bash
ollama run gemma:2b
```

### 4. Start the App

```bash
streamlit run app.py
```

---

This application runs fully **offline** and does not require any API keys or external LLM services.
