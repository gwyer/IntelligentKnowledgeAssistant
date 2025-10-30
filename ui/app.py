# app.py - Streamlit UI for RAG Knowledge Assistant

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path to import from src
sys.path.append(str(Path(__file__).parent.parent))

from src.generator import generate_answer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Intelligent Knowledge Assistant",
    page_icon="🧠",
    layout="wide"
)

# Title and description
st.title("🧠 Intelligent Knowledge Assistant")
st.markdown("Ask questions about your documents using RAG (Retrieval-Augmented Generation)")

# Initialize conversational memory in session state
st.session_state.setdefault("history", [])

# Sidebar for configuration and history
with st.sidebar:
    st.header("⚙️ Configuration")

    # Show current LLM provider
    import os
    llm_provider = os.getenv("LLM_PROVIDER", "ollama")
    st.info(f"**LLM Provider:** {llm_provider.upper()}")

    if llm_provider == "ollama":
        ollama_model = os.getenv("OLLAMA_MODEL", "deepseek-r1:8b")
        st.info(f"**Model:** {ollama_model}")

    st.divider()

    # Model Parameters
    st.header("🎛️ Model Parameters")

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.2,
        step=0.1,
        help="Controls randomness: Lower = more focused, Higher = more creative"
    )

    top_p = st.slider(
        "Top-p (nucleus sampling)",
        min_value=0.0,
        max_value=1.0,
        value=0.9,
        step=0.05,
        help="Cumulative probability cutoff for token selection"
    )

    top_k = st.slider(
        "Top-k",
        min_value=1,
        max_value=100,
        value=40,
        step=1,
        help="Number of highest probability tokens to consider"
    )

    st.divider()

    # Clear history button
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state["history"] = []
            st.rerun()

    # Export history button
    with col2:
        if st.session_state["history"] and st.button("💾 Export", use_container_width=True):
            # Create formatted text export
            import json
            from datetime import datetime

            export_data = {
                "export_date": datetime.now().isoformat(),
                "total_conversations": len(st.session_state["history"]),
                "conversations": st.session_state["history"]
            }

            # Convert to JSON
            json_str = json.dumps(export_data, indent=2)

            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"conversation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

            # Also offer markdown format
            md_export = f"# Conversation History\n\n**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            md_export += f"**Total Conversations:** {len(st.session_state['history'])}\n\n---\n\n"

            for i, conv in enumerate(st.session_state["history"], 1):
                md_export += f"## Conversation {i}\n\n"
                md_export += f"**Question:**\n{conv['user']}\n\n"
                md_export += f"**Answer:**\n{conv['answer']}\n\n"
                if conv.get("token_usage"):
                    tokens = conv["token_usage"]
                    md_export += f"*Tokens: Prompt={tokens.get('prompt_tokens', 0)}, "
                    md_export += f"Completion={tokens.get('completion_tokens', 0)}, "
                    md_export += f"Total={tokens.get('total_tokens', 0)}*\n\n"
                md_export += "---\n\n"

            st.download_button(
                label="📥 Download Markdown",
                data=md_export,
                file_name=f"conversation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )

    # Show conversation history count
    st.metric("Conversations", len(st.session_state["history"]))

    st.divider()

    # Conversation history
    if st.session_state["history"]:
        st.header("📜 History")
        for i, conv in enumerate(reversed(st.session_state["history"])):
            conv_num = len(st.session_state['history']) - i
            question_preview = conv['user'][:50] + "..." if len(conv['user']) > 50 else conv['user']
            with st.expander(f"Q{conv_num}: {question_preview}"):
                st.markdown(f"**Question:**")
                st.write(conv['user'])
                st.markdown(f"**Answer:**")
                st.write(conv['answer'])  # Full answer, no truncation
                # Show token usage if available
                if conv.get("token_usage"):
                    tokens = conv["token_usage"]
                    st.caption(f"(Tokens: Prompt={tokens.get('prompt_tokens', 0)}, "
                             f"Completion={tokens.get('completion_tokens', 0)}, "
                             f"Total={tokens.get('total_tokens', 0)})")

# Main chat interface
st.header("💬 Ask a Question")

# Query input
query = st.text_input(
    "Enter your question:",
    placeholder="What is required for the interview?",
    key="query_input"
)

# Submit button
if st.button("🔍 Ask", type="primary") or (query and st.session_state.get("auto_submit", False)):
    if query:
        with st.spinner("🤔 Thinking..."):
            try:
                # Generate answer with custom parameters
                result = generate_answer(
                    query,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k
                )
                answer = result["answer"]
                sources = result["source_documents"]
                token_usage = result.get("token_usage", {})

                # Add to conversation history with token usage
                st.session_state["history"].append({
                    "user": query,
                    "answer": answer,
                    "token_usage": token_usage
                })

                # Display answer
                st.success("✅ Answer Generated")
                st.markdown("### 🧠 Answer:")
                st.markdown(answer)

                # Display token usage in different color with parentheses
                if token_usage:
                    prompt_tokens = token_usage.get('prompt_tokens', 0)
                    completion_tokens = token_usage.get('completion_tokens', 0)
                    total_tokens = token_usage.get('total_tokens', 0)

                    st.markdown(
                        f"<p style='color: #888; font-size: 0.9em;'>"
                        f"(Tokens used: Prompt={prompt_tokens}, Completion={completion_tokens}, Total={total_tokens})"
                        f"</p>",
                        unsafe_allow_html=True
                    )

                # Display sources
                st.markdown("### 📄 Sources:")
                for i, doc in enumerate(sources, 1):
                    source = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", "N/A")
                    with st.expander(f"Source {i}: {source} (Page {page})"):
                        st.text(doc.page_content)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)
    else:
        st.warning("⚠️ Please enter a question.")

# Display recent conversations
if st.session_state["history"]:
    st.divider()
    st.header("🕒 Recent Conversations")

    # Show last 3 conversations with full text in expandable sections
    for i, conv in enumerate(reversed(st.session_state["history"][:3])):
        conv_num = len(st.session_state["history"]) - i

        # Create preview
        question_preview = conv['user'][:100] + "..." if len(conv['user']) > 100 else conv['user']
        answer_preview = conv['answer'][:200] + "..." if len(conv['answer']) > 200 else conv['answer']

        with st.container():
            st.markdown(f"### Conversation {conv_num}")

            # Question (always show full)
            st.markdown(f"**Question:** {conv['user']}")

            # Answer with expandable full text
            if len(conv['answer']) > 200:
                with st.expander("📖 Show Full Answer"):
                    st.markdown(conv['answer'])
                st.markdown(f"**Answer Preview:** {answer_preview}")
            else:
                st.markdown(f"**Answer:** {conv['answer']}")

            # Show token usage
            if conv.get("token_usage"):
                tokens = conv["token_usage"]
                st.markdown(
                    f"<p style='color: #888; font-size: 0.85em;'>"
                    f"(Tokens: Prompt={tokens.get('prompt_tokens', 0)}, "
                    f"Completion={tokens.get('completion_tokens', 0)}, "
                    f"Total={tokens.get('total_tokens', 0)})</p>",
                    unsafe_allow_html=True
                )
            st.divider()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <small>Powered by LangChain, FAISS, and Local LLMs | Built with Streamlit</small>
    </div>
    """,
    unsafe_allow_html=True
)
