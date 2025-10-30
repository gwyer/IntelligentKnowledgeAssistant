# generator.py — Query and Answer Generation

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os


def get_llm(temperature=0.2, top_p=0.9, top_k=40):
    """Get the appropriate LLM based on environment configuration with custom parameters."""
    llm_provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if llm_provider == "openai":
        print(f"Using OpenAI GPT-4 (temp={temperature}, top_p={top_p})...")
        return ChatOpenAI(
            temperature=temperature,
            model_kwargs={"top_p": top_p}
        )
    elif llm_provider == "ollama":
        ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        print(f"Using Ollama ({ollama_model}) at {ollama_base_url} (temp={temperature}, top_p={top_p}, top_k={top_k})...")
        return ChatOllama(
            model=ollama_model,
            base_url=ollama_base_url,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {llm_provider}. Use 'openai' or 'ollama'.")


def load_vector_store(index_path: str = "faiss_index"):
    """Load FAISS vector store from local index."""
    # Use the same free local embeddings model with offline mode
    # local_files_only=True prevents any internet access
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={
            'device': 'cpu',
            'local_files_only': True
        },
        encode_kwargs={'normalize_embeddings': True}
    )
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def generate_answer(query: str, temperature=0.2, top_p=0.9, top_k=40):
    """Generates an answer from the RAG pipeline for a given query with custom parameters."""
    db = load_vector_store()
    # Increased k from 3 to 5 for better context, especially for accounting queries
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # Get LLM based on environment configuration (OpenAI or Ollama) with custom parameters
    llm = get_llm(temperature=temperature, top_p=top_p, top_k=top_k)

    # Create accounting-aware prompt template
    prompt = ChatPromptTemplate.from_template("""You are an intelligent accounting assistant analyzing financial documents and general ledgers.

Context: {context}

Question: {question}

Instructions:
- Carefully examine any tables, account names, dates, amounts, and transactions in the context
- If the question involves calculations or summations, show your work step by step
- Reference specific account numbers, account names, dates, and dollar amounts from the source material
- If you see tabular data, interpret column headers and row values accurately
- Distinguish between debits and credits, assets and liabilities, income and expenses
- If the requested information is not in the context or is unclear, explicitly state this
- Format monetary amounts clearly (e.g., $1,234.56)

Answer:""")

    # Create RAG chain using LCEL (LangChain Expression Language)
    # We'll use the LLM directly to capture token usage
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    # Get retrieved documents for source display
    retrieved_docs = retriever.invoke(query)

    # Invoke the chain and capture full response
    response = rag_chain.invoke(query)

    # Extract answer text
    if hasattr(response, 'content'):
        answer = response.content
    else:
        answer = str(response)

    # Extract token usage if available
    token_usage = {}
    if hasattr(response, 'response_metadata'):
        metadata = response.response_metadata
        if 'token_usage' in metadata:
            token_usage = metadata['token_usage']
        elif 'usage' in metadata:
            token_usage = metadata['usage']
        # For Ollama
        if 'eval_count' in metadata:
            token_usage = {
                'prompt_tokens': metadata.get('prompt_eval_count', 0),
                'completion_tokens': metadata.get('eval_count', 0),
                'total_tokens': metadata.get('prompt_eval_count', 0) + metadata.get('eval_count', 0)
            }

    print("\n🔎 Query:", query)
    print("🧠 Answer:", answer)
    print("📄 Sources:")
    for doc in retrieved_docs:
        print(f"- {doc.metadata.get('source', 'Unknown')} (page {doc.metadata.get('page', 'N/A')})")
    if token_usage:
        print(f"🔢 Tokens: {token_usage}")

    return {
        "query": query,
        "answer": answer,
        "source_documents": retrieved_docs,
        "token_usage": token_usage
    }


if __name__ == "__main__":
    load_dotenv()
    user_query = input("Enter your question: ")
    generate_answer(user_query)