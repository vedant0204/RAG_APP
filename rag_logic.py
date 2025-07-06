import os
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

# Configuration
CHROMA_PATH = "chroma"
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Util for vector DB
def get_vectordb():
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=EMBEDDING_MODEL)

# Loader function
def load_document(path):
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".pdf":
        return PyPDFLoader(path).load()
    elif ext == ".txt":
        return TextLoader(path).load()
    elif ext == ".md":
        return UnstructuredMarkdownLoader(path).load()
    elif ext in [".doc", ".docx"]:
        return UnstructuredWordDocumentLoader(path).load()
    else:
        raise ValueError("Unsupported file type")

# Split, embed, and persist document
def process_and_store_doc(path):
    docs = load_document(path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    db = get_vectordb()
    db.add_documents(chunks)
    db.persist()

# Helper to match words
def contains_whole_word(word, text):
    return re.search(rf'\b{re.escape(word)}\b', text) is not None

# Main QA function
def answer_question(query):
    query_lower = query.lower().strip()

    # Predefined responses
    if any(contains_whole_word(greet, query_lower) for greet in ["hi", "hello", "hey"]):
        return "Hello. How may I assist you today?", []

    elif contains_whole_word("how are you", query_lower):
        return "I'm functioning smoothly. How can I help you today?", []

    elif any(contains_whole_word(thx, query_lower) for thx in ["thank you", "thanks"]):
        return "You're most welcome!", []

    elif contains_whole_word("tell me a joke", query_lower):
        return "Why do programmers hate nature? Too many bugs!", []

    elif contains_whole_word("do you sleep", query_lower):
        return "I never sleep. I'm always on call for your questions!", []

    elif any(contains_whole_word(phrase, query_lower) for phrase in ["great job", "well done"]):
        return "Thank you! I strive to be helpful.", []

    elif any(contains_whole_word(phrase, query_lower) for phrase in ["are you real", "are you a human"]):
        return "I’m not human — just an AI assistant trained to help you understand documents.", []

    elif any(contains_whole_word(phrase, query_lower) for phrase in ["who are you", "your name"]):
        return "I’m a document assistant built to answer questions based on uploaded files.", []

    elif any(contains_whole_word(phrase, query_lower) for phrase in ["what can you do", "help"]):
        return "You may upload a document, and then ask any question related to its content.", []

    elif any(contains_whole_word(phrase, query_lower) for phrase in ["bye", "goodbye"]):
        return "Goodbye. Feel free to return anytime.", []

    elif contains_whole_word("what is your purpose", query_lower):
        return "My purpose is to assist you with questions about your documents. Just upload a file and ask away!", []

    # === RAG Logic ===
    vectordb = get_vectordb()
    retriever = vectordb.as_retriever(search_type="similarity", k=6)

    docs_with_scores = retriever.vectorstore.similarity_search_with_relevance_scores(query, k=3)
    RELEVANCE_THRESHOLD = 0.2
    if all(score < RELEVANCE_THRESHOLD for _, score in docs_with_scores):
        return ("I'm sorry, I couldn't find relevant information. Please rephrase your question or upload a more detailed document.", [])

    # Using Ollama model
    llm = Ollama(model="gemma:2b")
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    result = qa(query)
    answer = result["result"]
    source_files = list({os.path.basename(doc.metadata.get("source", "Unknown")) for doc in result["source_documents"]})
    source_str = ", ".join(source_files)
    return f"{answer}\n\n**Sources:** {source_str}", source_files
            
    if not answer.strip() or answer.strip().lower() in ["i don't know", "no information available", "not sure"]:
        return (
            "I'm sorry, I couldn't find an answer based on the uploaded document. Try rephrasing your question.",
            []
        )

    return answer, sources
