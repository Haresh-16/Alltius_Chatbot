import streamlit as st
import logging
from datetime import datetime
from rag_system import initialize_rag_system

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="🤖 Multi-Agent RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "system_logs" not in st.session_state:
    st.session_state.system_logs = []

def initialize_system():
    """Initialize the RAG system using Streamlit secrets"""
    try:
        with st.spinner("🔧 Initializing Multi-Agent RAG System..."):
            rag_system = initialize_rag_system()
            if rag_system:
                st.session_state.rag_system = rag_system
                st.success("✅ RAG system initialized successfully!")
                return True
            else:
                st.error("❌ Failed to initialize RAG system. Please check your secrets configuration.")
                return False
    except Exception as e:
        st.error(f"❌ Error initializing system: {str(e)}")
        logger.error(f"System initialization error: {e}")
        return False

def display_agent_logs():
    """Display internal agent logs in the sidebar"""
    if st.session_state.rag_system and hasattr(st.session_state.rag_system, 'internal_logs'):
        if st.session_state.rag_system.internal_logs:
            st.sidebar.subheader("🔍 Agent Activity Logs")
            for log in st.session_state.rag_system.internal_logs[-10:]:  # Show last 10 logs
                timestamp = log.get('timestamp', '')
                agent = log.get('agent', '')
                action = log.get('action', '')
                st.sidebar.text(f"[{timestamp}] {agent}: {action[:50]}...")

def main():
    # Header
    st.title("🤖 Multi-Agent RAG Chatbot")
    st.markdown("**Specialized in Angel One Support & Health Insurance Plans**")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This chatbot uses a **Multi-Agent RAG Architecture** with:
        - 🧠 **Head Agent**: Orchestrates workflow
        - 🔍 **Query Agent**: Analyzes questions  
        - 📚 **Retriever Agent**: Finds relevant docs
        - 🎯 **Reranking Agent**: Ranks by relevance
        - 💬 **Answering Agent**: Generates responses
        
        **Powered by:**
        - 🤖 Mistral-7B-Instruct-v0.3
        - 📊 Pinecone Vector Database
        - 🔗 LangChain Framework
        """)
        
        # System status
        if st.session_state.rag_system is None:
            st.warning("⚠️ System not initialized")
            if st.button("🔧 Initialize System"):
                initialize_system()
        else:
            st.success("✅ System Ready")
            
            # Clear conversation button
            if st.button("🗑️ Clear Conversation"):
                st.session_state.messages = []
                st.session_state.system_logs = []
                if st.session_state.rag_system:
                    st.session_state.rag_system.memory.clear()
                st.rerun()
        
        st.divider()
        
        # Display agent logs
        display_agent_logs()
    
    # Main chat interface
    if st.session_state.rag_system is None:
        st.info("👆 Please initialize the system using the sidebar button to start chatting.")
        st.markdown("""
        ### 🚀 Getting Started
        
        1. Click **"🔧 Initialize System"** in the sidebar
        2. Wait for the system to load (this may take a moment)
        3. Start asking questions about:
           - 📈 Angel One trading platform
           - 🏥 Health insurance plans
           
        ### 💡 Example Questions
        - "How do I add funds to my Angel One account?"
        - "What are the eligibility requirements for health insurance?"
        - "How do I place a buy order on Angel One?"
        - "What is the waiting period for pre-existing conditions?"
        """)
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show contexts for assistant messages if available
            if message["role"] == "assistant" and "contexts" in message:
                with st.expander("📚 View Sources", expanded=False):
                    for i, context in enumerate(message["contexts"], 1):
                        st.text_area(
                            f"Source {i}",
                            value=context[:500] + "..." if len(context) > 500 else context,
                            height=100,
                            disabled=True
                        )
    
    # Chat input
    if prompt := st.chat_input("Ask me about Angel One or health insurance plans..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤖 Processing your question..."):
                try:
                    # Get response from RAG system
                    response = st.session_state.rag_system.query(prompt)
                    answer = response.get("answer", "I apologize, but I couldn't generate a response.")
                    contexts = response.get("contexts", [])
                    
                    # Display the answer
                    st.markdown(answer)
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "contexts": contexts
                    })
                    
                    # Show contexts
                    if contexts:
                        with st.expander("📚 View Sources", expanded=False):
                            for i, context in enumerate(contexts, 1):
                                st.text_area(
                                    f"Source {i}",
                                    value=context[:500] + "..." if len(context) > 500 else context,
                                    height=100,
                                    disabled=True
                                )
                    
                    # Refresh agent logs in sidebar
                    st.rerun()
                    
                except Exception as e:
                    error_msg = f"I apologize, but I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    logger.error(f"Query processing error: {e}")

if __name__ == "__main__":
    main() 