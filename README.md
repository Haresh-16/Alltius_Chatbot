

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

## 📁 Project Structure

### 🎯 **Core Application Files**
- **`streamlit_app.py`** - Main Streamlit web application with chat interface and real-time agent logs
- **`rag_system.py`** - Multi-Agent RAG system implementation with all five specialized agents
- **`requirements.txt`** - Python dependencies and package versions for the project

### 🗄️ **Data Processing Pipeline**

The data processing pipeline is a sophisticated multi-stage system that transforms raw documents and web content into structured, searchable knowledge chunks optimized for RAG retrieval. Here's how each stage works:

#### **📄 Stage 1: Document Extraction (`doc_extraction.py`)**

**Purpose**: Extracts and preprocesses content from health insurance policy documents (PDF/DOCX)

**Technical Implementation**:
- **PDF Processing**: Uses PyMuPDF (fitz) to extract text and tables from insurance PDFs
  - Processes first 5 pages of each document for efficiency
  - Extracts table data with proper header-value mapping
  - Identifies and extracts text blocks while avoiding table overlap
  - Creates structured chunks with metadata (page number, file source, plan name)

- **DOCX Processing**: Uses python-docx to extract paragraph content
  - Processes all paragraphs in the document
  - Maintains document structure and formatting context

- **Advanced Cleaning & Filtering**:
  - **Boilerplate Removal**: Eliminates page numbers, copyright notices, disclaimers
  - **Length Filtering**: Removes chunks with fewer than 10 words
  - **Duplicate Detection**: Uses Levenshtein distance (95% similarity threshold) to remove near-duplicates
  - **Content Normalization**: Standardizes whitespace and formatting

**Output**: 
- `all_pdf_chunks_cleaned.json` - ~524 cleaned chunks from insurance PDFs
- `all_docx_chunks_cleaned.json` - ~82 cleaned chunks from DOCX files

#### **🌐 Stage 2: Web Content Scraping (`web_scraper.py`)**

**Purpose**: Scrapes Angel One support pages for trading platform FAQs and guides

**Technical Implementation**:
- **Dynamic Content Handling**: Uses Playwright with Chromium for JavaScript-rendered content
  - Waits for `networkidle` state to ensure complete page loading
  - Handles single-page applications and dynamic content updates

- **Intelligent Crawling**:
  - Recursive link discovery within the `/support/` domain
  - Automatic URL normalization and duplicate prevention
  - Excludes Hindi language pages (`/hindi/` paths)
  - Implements polite crawling with configurable delays (1.5s default)

- **Content Extraction Strategy**:
  - Targets main content areas (`article-content`, `main`, `body` fallbacks)
  - Removes navigation, footer, scripts, and other non-content elements
  - Extracts Q&A pairs by identifying question-answer patterns
  - Creates structured chunks with rich metadata (URL, page title, section context)

**Output**: `angelone_faqs_chunks.json` - Raw scraped content ready for post-processing

#### **🧹 Stage 3: Post-Processing & Refinement (`angelone_faqs_postproc.py`)**

**Purpose**: Advanced cleaning and quality enhancement of scraped Angel One content

**Technical Implementation**:
- **FAQ-Specific Pattern Removal**: Eliminates web-specific noise:
  - Feedback prompts ("Was this article helpful?", "Yes/No" buttons)
  - Navigation elements ("Related articles", "Back to top")
  - Social sharing widgets ("Share this article", social media links)
  - Generic website elements (copyright, privacy policy, footers)
  - Contact/support prompts ("Still have questions?")

- **Advanced Deduplication**:
  - **Exact Duplicate Removal**: Eliminates identical content strings
  - **Near-Duplicate Detection**: Uses fuzzy matching (90% similarity threshold)
  - **Content Length Filtering**: Removes chunks shorter than 50 words

- **Quality Enhancement**:
  - Text normalization and whitespace standardization
  - Content validation and structure preservation
  - Metadata enrichment and standardization

**Output**: `cleaned_angelone_faqs_chunks.json` - ~632 high-quality, deduplicated chunks

#### **🗃️ Stage 4: Vector Database Setup (`setup_pinecone.py`)**

**Purpose**: Initialize and configure the Pinecone vector database infrastructure

**Technical Implementation**:
- **Environment Validation**: Checks for required API keys and data availability
- **Index Configuration**: 
  - Creates Pinecone index with 384-dimensional vectors (all-MiniLM-L6-v2 embeddings)
  - Uses cosine similarity metric for semantic search
  - Implements serverless architecture (AWS us-east-1)

- **Namespace Planning**: Maps JSON files to organized namespaces:
  - `all-pdf-chunks-cleaned` → Insurance PDF content
  - `all-docx-chunks-cleaned` → Insurance DOCX content  
  - `cleaned-angelone-faqs-chunks` → Angel One support content

**Output**: Configured Pinecone index ready for data loading

#### **🚀 Stage 5: Vector Embedding & Loading (`pinecone_loader.py`)**

**Purpose**: Convert processed text chunks into searchable vector embeddings and load into Pinecone

**Technical Implementation**:
- **Embedding Generation**:
  - Uses SentenceTransformers `all-MiniLM-L6-v2` model (384 dimensions)
  - Optimized for semantic similarity and fast inference
  - Batch processing with progress tracking for efficiency

- **Vector Preparation**:
  - Generates unique IDs for each chunk (`namespace_uuid_index`)
  - Creates comprehensive metadata:
    - Full content (truncated to 1000 chars for metadata limits)
    - Source information (file, page, URL)
    - Content statistics (length, type)
    - Namespace organization for domain separation

- **Batch Upload Strategy**:
  - Uploads vectors in batches of 100 for optimal performance
  - Implements error handling and retry logic
  - Progress tracking with tqdm for user feedback
  - Separate namespace isolation for different content types

**Output**: Fully populated Pinecone vector database with three specialized namespaces

#### **🔄 Pipeline Flow Summary**

```
Raw Documents → Extract & Clean → Web Scraping → Post-Process → Vector DB Setup → Embed & Load
     ↓              ↓                ↓              ↓              ↓              ↓
Insurance PDFs → Text Chunks → Angel One FAQs → Quality Chunks → Pinecone Index → Searchable Vectors
Insurance DOCX                                                                        
```

**Key Pipeline Features**:
- **Multi-Source Integration**: Handles both document extraction and web scraping
- **Quality Assurance**: Multiple cleaning and filtering stages ensure high-quality data
- **Scalable Architecture**: Modular design allows easy addition of new data sources
- **Semantic Organization**: Namespace separation enables targeted retrieval
- **Metadata Preservation**: Maintains source attribution throughout the pipeline
- **Error Resilience**: Comprehensive error handling and logging at each stage

### 🧪 **Evaluation & Testing**
- **`eval_rag.py`** - RAGAs evaluation framework to assess system performance with metrics

### 🐳 **Deployment Configuration**
- **`Dockerfile`** - Docker container configuration for deployment
- **`.dockerignore`** - Specifies files to exclude from Docker builds
- **`.gitignore`** - Git version control exclusions

### 📊 **Data Storage**
- **`jsons_from_sources/`** - Directory containing processed JSON chunks from all data sources
  - `all_pdf_chunks_cleaned.json` - Processed health insurance PDF content
  - `all_docx_chunks_cleaned.json` - Processed health insurance DOCX content  
  - `cleaned_angelone_faqs_chunks.json` - Processed Angel One FAQ content
- **`Insurance PDFs/`** - Raw health insurance policy documents (PDF/DOCX format)
- **`Insurance PDFs.zip`** - Compressed archive of insurance documents

### ⚙️ **Development Files**
- **`README.md`** - Project documentation and setup instructions
- **`venv/`** - Python virtual environment (excluded from version control)
- **`__pycache__/`** - Python bytecode cache (excluded from version control)
- **`.git/`** - Git repository metadata

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

## 🧪 Future Work

To complete the evaluation script ```eval_rag.py``` to assess system performance:

**Evaluation Metrics:**
- **Faithfulness**: Answer accuracy to source content
- **Answer Relevancy**: Response relevance to question  
- **Context Precision**: Quality of retrieved context
- **Context Recall**: Coverage of relevant information

## 🙏 Acknowledgments

- **Mistral AI** for the open-source LLM
- **Pinecone** for vector database infrastructure  
- **Hugging Face** for model hosting and spaces
- **LangChain** for the RAG framework
- **Streamlit** for the intuitive UI framework
