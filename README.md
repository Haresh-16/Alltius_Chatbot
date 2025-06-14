---
title: Multi-Agent RAG Chatbot
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: streamlit
app_file: streamlit_app.py
pinned: false
---

# 🤖 Multi-Agent RAG Chatbot

A sophisticated chatbot system built with **Multi-Agent RAG Architecture** that specializes in:
- 📈 **Angel One Support**: Trading platform assistance and FAQ
- 🏥 **Health Insurance Plans**: Insurance-related queries and guidance

## 🏗️ Multi-Agent Architecture

This system implements a sophisticated **Multi-Agent RAG (Retrieval-Augmented Generation)** architecture with specialized agents:

### 🧠 **Head Agent** 
- **Role**: Central orchestrator and coordinator
- **Function**: Manages workflow between all other agents
- **Capability**: Routes queries, makes decisions, and ensures smooth operation

### 🔍 **Query Agent**
- **Role**: Query analysis and categorization  
- **Function**: Analyzes incoming user questions to determine intent
- **Capability**: Classifies queries as Angel One support, health insurance, or irrelevant

### 📚 **Retriever Agent**
- **Role**: Document retrieval specialist
- **Function**: Searches vector database for relevant documents
- **Capability**: Performs semantic search across different namespaces

### 🎯 **Reranking Agent**  
- **Role**: Relevance optimization specialist
- **Function**: Reorders retrieved documents by relevance to query
- **Capability**: Uses LLM-based reranking for improved context selection

### 💬 **Answering Agent**
- **Role**: Response generation specialist  
- **Function**: Generates final answers using retrieved context
- **Capability**: Produces human-like responses with source attribution

## 🔧 Environment Variables Required

### For Hugging Face Spaces Deployment

This application requires the following environment variables to be set in your Hugging Face Space settings:

1. **PINECONE_API_KEY**: Your Pinecone API key
   - Get it from [Pinecone Console](https://app.pinecone.io/)
   - Used for vector database operations

2. **HUGGING_FACE_HUB_TOKEN**: Your Hugging Face API token
   - Get it from [Hugging Face Settings](https://huggingface.co/settings/tokens)
   - Required for Mistral-7B-Instruct-v0.3 model access

### For Local Development

Create a `.env` file in the project root:

```bash
PINECONE_API_KEY=your_pinecone_key_here
HUGGING_FACE_HUB_TOKEN=your_hf_token_here
# Optional: For evaluation only
OPENAI_API_KEY=your_openai_key_here
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Pinecone API key
- Hugging Face Hub token

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd multi-agent-rag-chatbot
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
# For local development
export PINECONE_API_KEY="your_key_here"
export HUGGING_FACE_HUB_TOKEN="your_token_here"

# For Hugging Face Spaces: Set these in Space settings
```

4. **Run the application**
```bash
streamlit run streamlit_app.py
```

## 🌐 Deployment

### Hugging Face Spaces

1. **Create a new Space** on [Hugging Face Spaces](https://huggingface.co/spaces)
2. **Choose Streamlit SDK**
3. **Upload your files** or connect your Git repository
4. **Set environment variables** in Space settings:
   - `PINECONE_API_KEY`
   - `HUGGING_FACE_HUB_TOKEN`
5. **Deploy** - Your space will automatically build and run

### Docker Deployment

```bash
# Build the Docker image
docker build -t rag-chatbot .

# Run with environment variables
docker run -p 7860:7860 \
  -e PINECONE_API_KEY="your_key" \
  -e HUGGING_FACE_HUB_TOKEN="your_token" \
  rag-chatbot
```

## 💡 Usage Examples

### Angel One Support
- "How do I add funds to my Angel One account?"
- "What are the brokerage charges for equity trading?"
- "How do I place a stop loss order?"
- "How to enable margin trading?"

### Health Insurance  
- "What are the eligibility requirements for health insurance?"
- "What is the waiting period for pre-existing conditions?"
- "How do I claim insurance for hospitalization?"
- "What is the difference between individual and family floater plans?"

## 🛠️ Technical Stack

- **🤖 LLM**: Mistral-7B-Instruct-v0.3 (HuggingFace)
- **🗄️ Vector Database**: Pinecone
- **🔗 Framework**: LangChain
- **🎨 UI**: Streamlit
- **📊 Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **🐳 Deployment**: Docker + Hugging Face Spaces

## 📊 Data Sources

### Angel One Support Data
- **Source**: Web scraping from Angel One support pages
- **Content**: FAQs, guides, trading procedures
- **Processing**: Cleaned and chunked JSON format

### Health Insurance Data  
- **Source**: Insurance policy documents (PDF/DOCX)
- **Content**: Policy terms, eligibility criteria, claim procedures
- **Processing**: Extracted and chunked with metadata

## 🔍 System Features

### 🎯 **Smart Query Classification**
- Automatically determines query type and relevance
- Routes to appropriate knowledge base
- Handles multi-turn conversations

### 📚 **Advanced Retrieval**
- Semantic search with vector embeddings
- Multi-namespace document retrieval  
- LLM-powered reranking for precision

### 💬 **Contextual Responses**
- Maintains conversation memory
- Source attribution and transparency
- Human-like response generation

### 🔧 **Real-time Monitoring**
- Agent activity logs in sidebar
- System status indicators
- Error handling and recovery

## 🧪 Evaluation

Run the evaluation script to assess system performance:

```bash
python eval_rag.py
```

**Evaluation Metrics:**
- **Faithfulness**: Answer accuracy to source content
- **Answer Relevancy**: Response relevance to question  
- **Context Precision**: Quality of retrieved context
- **Context Recall**: Coverage of relevant information

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Mistral AI** for the open-source LLM
- **Pinecone** for vector database infrastructure  
- **Hugging Face** for model hosting and spaces
- **LangChain** for the RAG framework
- **Streamlit** for the intuitive UI framework
