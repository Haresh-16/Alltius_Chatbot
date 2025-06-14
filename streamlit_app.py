import streamlit as st
import logging
from datetime import datetime
from rag_system import initialize_rag_system
import os

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
if "conversation_logs" not in st.session_state:
    st.session_state.conversation_logs = []
if "current_turn_logs" not in st.session_state:
    st.session_state.current_turn_logs = []

def initialize_system():
    """Initialize the RAG system"""
    try:
        with st.spinner("🔧 Initializing Multi-Agent RAG System..."):
            rag_system = initialize_rag_system()
            if rag_system:
                st.session_state.rag_system = rag_system
                st.success("✅ RAG system initialized successfully!")
                return True
            else:
                st.error("❌ Failed to initialize RAG system. Please check your environment variables.")
                return False
    except Exception as e:
        st.error(f"❌ Error initializing system: {str(e)}")
        logger.error(f"System initialization error: {e}")
        return False

def display_agent_logs():
    """Display organized agent logs by conversation turn"""
    st.sidebar.subheader("🔍 Agent Activity Logs")
    
    if not st.session_state.conversation_logs:
        st.sidebar.info("No conversation logs yet. Start chatting to see agent activity!")
        return
    
    # Show logs for each conversation turn
    for turn_idx, turn_data in enumerate(reversed(st.session_state.conversation_logs), 1):
        user_query = turn_data.get('user_query', 'Unknown query')
        logs = turn_data.get('logs', [])
        timestamp = turn_data.get('timestamp', '')
        
        # Create collapsible section for each turn
        with st.sidebar.expander(f"🔄 Turn {len(st.session_state.conversation_logs) - turn_idx + 1}: {user_query[:30]}..." if len(user_query) > 30 else f"🔄 Turn {len(st.session_state.conversation_logs) - turn_idx + 1}: {user_query}", expanded=(turn_idx == 1)):
            st.caption(f"⏰ {timestamp}")
            
            if not logs:
                st.caption("No detailed logs for this turn")
                continue
            
            # Group logs by agent
            agent_logs = {}
            for log in logs:
                agent = log.get('agent', 'Unknown')
                if agent not in agent_logs:
                    agent_logs[agent] = []
                agent_logs[agent].append(log)
            
            # Display logs by agent with proper formatting
            agent_order = ['Head Agent', 'Query Agent', 'Retriever Agent', 'Reranking Agent', 'Answering Agent', 'System']
            
            for agent in agent_order:
                if agent in agent_logs:
                    # Agent header with emoji
                    if agent == "Head Agent":
                        st.markdown(f"🧠 **{agent}**")
                    elif agent == "Query Agent":
                        st.markdown(f"🔍 **{agent}**")
                    elif agent == "Retriever Agent":
                        st.markdown(f"📚 **{agent}**")
                    elif agent == "Reranking Agent":
                        st.markdown(f"🎯 **{agent}**")
                    elif agent == "Answering Agent":
                        st.markdown(f"💬 **{agent}**")
                    else:
                        st.markdown(f"⚙️ **{agent}**")
                    
                    # Show agent's activities
                    for log in agent_logs[agent]:
                        action = log.get('action', '')
                        log_time = log.get('timestamp', '')
                        st.caption(f"  └─ [{log_time}] {action}")
                    
                    st.markdown("---")
            
            # Show any remaining agents not in the order
            for agent, logs_list in agent_logs.items():
                if agent not in agent_order:
                    st.markdown(f"⚙️ **{agent}**")
                    for log in logs_list:
                        action = log.get('action', '')
                        log_time = log.get('timestamp', '')
                        st.caption(f"  └─ [{log_time}] {action}")
                    st.markdown("---")

def capture_turn_logs():
    """Capture logs for the current conversation turn"""
    if st.session_state.rag_system and hasattr(st.session_state.rag_system, 'internal_logs'):
        # Get new logs since last capture
        current_logs = st.session_state.rag_system.internal_logs.copy()
        new_logs = current_logs[len(st.session_state.current_turn_logs):]
        return new_logs
    return []

def start_new_turn(user_query: str):
    """Start tracking a new conversation turn"""
    st.session_state.current_turn_logs = []
    if st.session_state.rag_system and hasattr(st.session_state.rag_system, 'internal_logs'):
        st.session_state.current_turn_logs = st.session_state.rag_system.internal_logs.copy()

def end_turn_and_save_logs(user_query: str):
    """End the current turn and save logs"""
    turn_logs = capture_turn_logs()
    
    turn_data = {
        'user_query': user_query,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'logs': turn_logs
    }
    
    st.session_state.conversation_logs.append(turn_data)
    
    # Keep only last 10 conversation turns to prevent memory issues
    if len(st.session_state.conversation_logs) > 10:
        st.session_state.conversation_logs = st.session_state.conversation_logs[-10:]

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
        
        # Environment Variables Status
        st.subheader("🔑 Environment Variables")
        pinecone_status = "✅" if os.getenv("PINECONE_API_KEY") else "❌"
        hf_status = "✅" if os.getenv("HUGGING_FACE_HUB_TOKEN") else "❌"
        st.text(f"{pinecone_status} PINECONE_API_KEY")
        st.text(f"{hf_status} HUGGING_FACE_HUB_TOKEN")
        
        st.divider()
        
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
                st.session_state.conversation_logs = []
                st.session_state.current_turn_logs = []
                if st.session_state.rag_system:
                    st.session_state.rag_system.memory.clear()
                    # Clear internal logs as well
                    st.session_state.rag_system.internal_logs = []
                st.rerun()
            
            # Clear logs button
            if st.button("🧹 Clear Logs"):
                st.session_state.conversation_logs = []
                st.session_state.current_turn_logs = []
                if st.session_state.rag_system:
                    st.session_state.rag_system.internal_logs = []
                st.rerun()
        
        st.divider()
        
        # Display agent logs
        display_agent_logs()
    
    # Main chat interface
    if st.session_state.rag_system is None:
        st.info("👆 Please initialize the system using the sidebar button to start chatting.")
        st.markdown("""
        ### 🚀 Getting Started
        
        1. **Set Environment Variables:**
           - `PINECONE_API_KEY`: Get from [Pinecone Console](https://app.pinecone.io/)
           - `HUGGING_FACE_HUB_TOKEN`: Get from [HF Settings](https://huggingface.co/settings/tokens)
        
        2. Click **"🔧 Initialize System"** in the sidebar
        3. Wait for the system to load (this may take a moment)
        4. Start asking questions about:
           - 📈 Angel One trading platform
           - 🏥 Health insurance plans
           
        ### 💡 Example Questions
        - "How do I add funds to my Angel One account?"
        - "What are the eligibility requirements for health insurance?"
        - "How do I place a buy order on Angel One?"
        - "What is the waiting period for pre-existing conditions?"
        
        ### 🔍 Agent Logs Feature
        Once you start chatting, you'll see **real-time agent activity logs** in the sidebar showing:
        - 🧠 Head Agent orchestrating the process
        - 🔍 Query Agent analyzing your question
        - 📚 Retriever Agent finding relevant documents  
        - 🎯 Reranking Agent optimizing results
        - 💬 Answering Agent generating responses
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
        # Start tracking new conversation turn
        start_new_turn(prompt)
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤖 Processing your question..."):
                try:
                    # Get response from RAG system (now returns a string directly)
                    response = st.session_state.rag_system.query(prompt)
                    
                    # Display the answer
                    st.markdown(response)
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response
                    })
                    
                    # End turn and save logs
                    end_turn_and_save_logs(prompt)
                    
                    # Refresh to update the interface and show new logs
                    st.rerun()
                    
                except Exception as e:
                    error_msg = f"I apologize, but I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    
                    # End turn and save logs even for errors
                    end_turn_and_save_logs(prompt)
                    logger.error(f"Query processing error: {e}")

if __name__ == "__main__":
    main() 