# api.py — FastAPI Endpoint for RAG Service

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from .generator import generate_answer
from dotenv import load_dotenv
import uvicorn

load_dotenv()

app = FastAPI(title="Intelligent Knowledge Assistant API",
              description="Ask questions over custom documents using RAG.",
              version="1.0.0")

# Allow local testing via CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ask")
def ask(
    query: str = Query(..., description="Your natural language question."),
    temperature: float = Query(0.2, description="Temperature (0.0-2.0): Controls randomness", ge=0.0, le=2.0),
    top_p: float = Query(0.9, description="Top-p (0.0-1.0): Nucleus sampling threshold", ge=0.0, le=1.0),
    top_k: int = Query(40, description="Top-k (1-100): Number of top tokens to consider", ge=1, le=100)
):
    """Endpoint that receives a query and returns an AI-generated answer with citations and token usage."""
    try:
        result = generate_answer(query, temperature=temperature, top_p=top_p, top_k=top_k)
        return {
            "query": query,
            "answer": result["answer"],
            "sources": [
                {
                    "file": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "N/A")
                }
                for doc in result["source_documents"]
            ],
            "token_usage": result.get("token_usage", {})
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
