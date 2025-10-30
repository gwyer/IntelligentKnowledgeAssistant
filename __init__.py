# Project: Intelligent Knowledge Assistant (RAG + Agents)
# Author: Chris Gwyer
# Purpose: Demonstrate practical RAG and Agentic AI skills for AI Engineer roles.

# Folder Structure:
# ├── data/                      # PDFs or text sources for ingestion
# ├── notebooks/                 # Experiments, embeddings visualization
# ├── src/
# │   ├── ingest.py              # Parse and chunk documents
# │   ├── embed_store.py         # Create and store embeddings (FAISS/Pinecone)
# │   ├── retriever.py           # Similarity search logic
# │   ├── generator.py           # LLM call for final answer generation
# │   ├── api.py                 # FastAPI app with /ask endpoint
# │   ├── evaluate.py            # Simple retrieval evaluation
# │   └── agent.py               # (Optional) Multi-tool LangChain agent
# ├── ui/
# │   └── app.py                 # Streamlit or React frontend (optional)
# ├── tests/                     # Unit tests
# ├── requirements.txt           # Dependencies
# ├── Dockerfile                 # For deployment
# ├── README.md                  # Project documentation
# └── diagram.png                # Architecture diagram

# Example code outline

# src/ingest.py
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_chunk(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    return chunks


# src/embed_store.py
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore


# src/retriever.py
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def retrieve(query: str, top_k=3):
    db = FAISS.load_local("faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    results = db.similarity_search(query, k=top_k)
    return results


# src/generator.py
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def generate_answer(query: str):
    db = FAISS.load_local("faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2)

    # Create prompt template
    prompt = ChatPromptTemplate.from_template("""Answer the question based on the context below:

Context: {context}

Question: {input}

Answer:""")

    # Create chains
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(db.as_retriever(), combine_docs_chain)

    response = retrieval_chain.invoke({"input": query})
    return response["answer"]


# src/api.py
from fastapi import FastAPI, Query
# from generator import generate_answer  # When src/ folder is created, use: from src.generator import generate_answer

app = FastAPI()

@app.get("/ask")
def ask(query: str = Query(...)):
    answer = generate_answer(query)
    return {"query": query, "answer": answer}


# README snippet
"""
### Intelligent Knowledge Assistant
A retrieval-augmented system that answers questions over custom documents.

**Core Stack:** LangChain, FAISS, OpenAI API, FastAPI, Streamlit.

**Key Features:**
- Document ingestion + chunking
- Embedding + vector storage
- Contextual retrieval
- LLM-based response generation
- REST API + optional UI
- Evaluation with RAGAS or basic accuracy metrics

**Run Locally:**
```
python src/ingest.py
python src/embed_store.py
uvicorn src.api:app --reload
```

**Next Steps:**
- Add caching for embeddings
- Integrate LangChain Agent for multi-step reasoning
- Deploy to Render or Hugging Face Spaces
"""
