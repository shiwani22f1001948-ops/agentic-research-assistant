from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdfs(folder):
    docs = []
    for pdf_path in Path(folder).glob("*.pdf"):
        loader = PyMuPDFLoader(str(pdf_path))
        docs.extend(loader.load())
    return docs


def split_docs(docs, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


if __name__ == "__main__":
    docs = load_pdfs("data/papers")
    chunks = split_docs(docs)

    print(f"Pages loaded: {len(docs)}")
    print(f"Chunks created: {len(chunks)}")
    print("\n--- Sample chunk ---")
    print(chunks[5].page_content)
    print("\nMetadata:", chunks[5].metadata)