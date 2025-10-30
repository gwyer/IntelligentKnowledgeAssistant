import streamlit as st
from generator import generate_answer
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Intelligent Knowledge Assistant", page_icon="🤖", layout="centered")

st.title("🤖 Intelligent Knowledge Assistant")
st.markdown("Ask questions about your uploaded documents using retrieval-augmented generation.")

# Display which LLM is being used
llm_provider = os.getenv("LLM_PROVIDER", "ollama")
llm_model = os.getenv("OLLAMA_MODEL", "llama3.2") if llm_provider == "ollama" else "gpt-4o"
st.caption(f"Using {llm_provider.upper()} ({llm_model})")

# Initialize session state for query
if 'query' not in st.session_state:
    st.session_state.query = ""

query = st.text_input("Enter your question:", value=st.session_state.query, key="query_input")

# Create columns for buttons
col1, col2 = st.columns([1, 5])

with col1:
    if st.button("Clear"):
        st.session_state.query = ""
        st.rerun()

with col2:
    ask_button = st.button("Ask")

if ask_button and query:
    with st.spinner('Thinking...'):
        try:
            # Call the generate_answer function from generator.py
            result = generate_answer(query)

            if result and "answer" in result:
                st.subheader("Answer")
                st.write(result["answer"])

                # Display sources from retrieved documents
                if result.get("source_documents"):
                    st.subheader("Sources")
                    for doc in result["source_documents"]:
                        source_file = doc.metadata.get('source', 'Unknown')
                        source_page = doc.metadata.get('page', 'N/A')
                        st.markdown(f"- **{source_file}** (page {source_page})")
            else:
                st.error("No answer returned.")
        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")
st.caption("Built with Streamlit + LangChain + Ollama (DeepSeek)")
