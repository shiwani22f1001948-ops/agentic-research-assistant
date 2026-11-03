from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from ingest import load_pdfs, split_docs

CHROMA_DIR = "chroma_db"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def build_vectorstore(chunks):
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=CHROMA_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )


def load_vectorstore():
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )


if __name__ == "__main__":
    if Path(CHROMA_DIR).exists():
        db = load_vectorstore()
        print("Loaded existing vector store")
    else:
        docs = load_pdfs("data/papers")
        chunks = split_docs(docs)
        db = build_vectorstore(chunks)
        print(f"Built vector store with {len(chunks)} chunks")

    query = "How does corrective RAG evaluate retrieved documents?"
    results = db.similarity_search_with_score(query, k=3)

    for i, (doc, score) in enumerate(results, 1):
        print(f"\n--- Result {i} | distance: {score:.3f} ---")
        print("Source:", doc.metadata["source"], "| Page:", doc.metadata["page"] + 1)
        print(doc.page_content[:300])