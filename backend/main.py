from pathlib import Path

import chromadb
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings

app = FastAPI(title="RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chroma_client = chromadb.PersistentClient(
    path=settings.CHROMA_PATH
)

collection = chroma_client.get_or_create_collection(
    name="documents"
)


def load_documents():
    docs_path = Path("/app/docs")

    documents = []

    if not docs_path.exists():
        return documents

    for file_path in docs_path.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        if text.strip():
            documents.append({
                "filename": file_path.name,
                "text": text
            })

    return documents


def ask_ollama(prompt):
    response = requests.post(
        f"{settings.OLLAMA_URL}/api/generate",
        json={
            "model": settings.MODEL_NAME,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["response"]


@app.get("/")
def root():
    return {
        "message": "RAG API running in Docker",
        "model": settings.MODEL_NAME,
        "max_results": settings.MAX_RESULTS,
        "debug": settings.DEBUG
    }


@app.get("/health")
def health():
    ollama_connected = False

    try:
        response = requests.get(
            f"{settings.OLLAMA_URL}/api/tags",
            timeout=3
        )

        ollama_connected = response.status_code == 200

    except requests.RequestException:
        pass

    return {
        "status": "healthy",
        "ollama": (
            "connected"
            if ollama_connected
            else "unavailable"
        ),
        "ollama_url": settings.OLLAMA_URL,
        "model": settings.MODEL_NAME,
        "documents": collection.count()
    }


@app.get("/stats")
def stats():
    return {
        "document_count": collection.count(),
        "max_results": settings.MAX_RESULTS,
        "confidence_threshold": settings.CONFIDENCE_THRESHOLD
    }


@app.post("/ingest")
def ingest():
    documents = load_documents()

    if not documents:
        return {
            "message": "No documents found.",
            "documents_ingested": 0
        }

    existing = collection.get()

    if existing["ids"]:
        collection.delete(
            ids=existing["ids"]
        )

    ids = []
    texts = []
    metadatas = []

    for index, document in enumerate(documents):
        ids.append(str(index))
        texts.append(document["text"])
        metadatas.append({
            "source": document["filename"]
        })

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas
    )

    return {
        "message": "Documents ingested successfully.",
        "documents_ingested": len(documents)
    }


@app.post("/ask")
def ask(question: dict):
    user_question = question.get("question", "").strip()

    if not user_question:
        return {
            "answer": "Please provide a question.",
            "sources": [],
            "confidence": "low"
        }

    if collection.count() == 0:
        return {
            "answer": "No documents have been ingested. Use POST /ingest first.",
            "sources": [],
            "confidence": "low"
        }

    results = collection.query(
        query_texts=[user_question],
        n_results=min(
            settings.MAX_RESULTS,
            collection.count()
        )
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_parts = []

    for document, metadata in zip(
        documents,
        metadatas
    ):
        context_parts.append(
            f"Source: {metadata['source']}\n"
            f"{document}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a helpful question-answering assistant.

Answer the user's question using only the provided context.

If the context does not contain enough information to answer
the question, say that the information is not available in
the provided documents.

Context:
{context}

Question:
{user_question}

Answer:
"""

    try:
        answer = ask_ollama(prompt)
    except requests.RequestException as error:
        return {
            "answer": f"Unable to contact Ollama: {error}",
            "sources": [
                metadata["source"]
                for metadata in metadatas
            ],
            "confidence": "low"
        }

    sources = list(
        dict.fromkeys(
            metadata["source"]
            for metadata in metadatas
        )
    )

    return {
        "answer": answer,
        "sources": sources,
        "confidence": "high"
    }