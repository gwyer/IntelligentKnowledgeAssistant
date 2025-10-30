# ingest.py — Document Loader and Chunker

import os

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk(pdf_dir: str = "data"):
    """Loads and chunks all PDF documents in the given directory.

    Uses PDFPlumberLoader for better table extraction, which is critical
    for accounting documents like general ledgers.
    """
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {pdf_dir}")

    # Increased chunk size for better context, especially for tables
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    all_chunks = []

    for file in pdf_files:
        path = os.path.join(pdf_dir, file)
        print(f"Loading {file} with table-aware parser...")
        # PDFPlumber preserves table structure better than PyPDF
        loader = PDFPlumberLoader(path)
        docs = loader.load()
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)


    print(f"✅ Processed {len(pdf_files)} PDF(s), total {len(all_chunks)} chunks.")
    return all_chunks

if __name__ == "__main__":
    chunks = load_and_chunk()
    print(f"First chunk sample:\n{chunks[0].page_content[:300]}...")