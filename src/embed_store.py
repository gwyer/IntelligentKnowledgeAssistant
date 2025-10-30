# embed_store.py - Embedding and Vector Store Creation

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from ingest import load_and_chunk
from dotenv import load_dotenv


def create_vector_store(chunks, index_path: str = "faiss_index"):
    """Creates and saves a FAISS vector store from document chunks."""
    # Use free local embeddings with offline mode
    # local_files_only=True prevents any internet access
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={
            'device': 'cpu',
            'local_files_only': True
        },
        encode_kwargs={'normalize_embeddings': True}
    )

    print(f"Creating embeddings for {len(chunks)} chunks...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    print(f"Saving vector store to {index_path}/")
    vectorstore.save_local(index_path)

    print("Vector store created and saved successfully!")
    return vectorstore


if __name__ == "__main__":
    load_dotenv()

    # Load and chunk documents
    print("Loading documents...")
    chunks = load_and_chunk()

    # Create and save vector store
    create_vector_store(chunks)
