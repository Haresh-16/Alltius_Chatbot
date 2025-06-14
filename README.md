---
title: Multi-Agent RAG Chatbot
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: streamlit
app_file: app.py
pinned: false
---

# 🤖 Multi-Agent RAG Chatbot

A sophisticated Multi-Agent Retrieval-Augmented Generation (RAG) chatbot designed for Angel One customer support and health insurance documentation. The system uses multiple specialized agents to process queries intelligently and provide accurate responses.

## 🚀 Features

- **Multi-Agent Architecture**: 5 specialized agents working together
  - **Head Agent**: Orchestrates the entire workflow
  - **Query Agent**: Analyzes and classifies user queries
  - **Retriever Agent**: Fetches relevant documents from Pinecone
  - **Reranking Agent**: Ranks documents by relevance
  - **Answering Agent**: Generates final responses

- **Intelligent Query Routing**: Automatically determines if queries are about:
  - Health plans and insurance eligibility
  - Angel One stock broking platform
  - Irrelevant topics (responds appropriately)

- **Advanced RAG Pipeline**: Uses Pinecone vector database with sentence transformers for semantic search

- **Powered by Mistral-7B-Instruct-v0.3**: Uses Hugging Face Inference API for high-quality responses

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **LLM**: Mistral-7B-Instruct-v0.3 via Hugging Face Inference API
- **Vector Database**: Pinecone
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Data Preprocessing**: PyMuPDF, python-docx, Levenshtein, BeautifulSoup, Playwright
- **Framework**: LangChain
- **Deployment**: Docker on Hugging Face Spaces

## 🔧 Environment Variables Required

This application requires the following secrets to be configured in the Hugging Face Space settings:

### Required Secrets for Hugging Face Spaces

1.  **PINECONE_API_KEY**: Your Pinecone API key
    -   Get it from [Pinecone Console](https://app.pinecone.io/)
    -   Used for vector database operations

2.  **HUGGING_FACE_HUB_TOKEN**: Your Hugging Face API token
    -   Get it from [Hugging Face Settings](https://huggingface.co/settings/tokens)
    -   Required for Mistral-7B-Instruct-v0.3 model access

### Optional Secrets (for evaluation only)

3.  **OPENAI_API_KEY**: Your OpenAI API key (optional)
    -   Only required if running the evaluation script with RAGAS
    -   Get it from [OpenAI Platform](https://platform.openai.com/api-keys)

### Setting Secrets in Hugging Face Spaces

1. Go to your Hugging Face Space settings
2. Navigate to the "Settings" tab
3. Scroll down to "Repository secrets"
4. Add each secret with the exact key names listed above
5. The application will automatically use these secrets when deployed

### Local Development

For local development, you can still use environment variables or a `.env` file:

```bash
# .env file
PINECONE_API_KEY=your_pinecone_key_here
HUGGING_FACE_HUB_TOKEN=your_hf_token_here
OPENAI_API_KEY=your_openai_key_here  # Optional
```

The application will automatically fall back to environment variables if Streamlit secrets are not available.

## 📊 Knowledge Base

The system has access to two distinct sources of truth:

-   **Health Insurance Documentation**: Eligibility criteria, coverage details, medical conditions from PDF and DOCX files.
-   **Angel One Support**: Web-scraped FAQs on trading, account management, IPOs, and mutual funds.

---

## ⚙️ Data Preprocessing Pipeline

Before being loaded into the vector database, source documents undergo a rigorous, source-specific preprocessing pipeline to ensure maximum data quality. Each step is handled by specialized Python scripts:

### 1. Health Insurance Documents (PDF & DOCX)
**📁 Handled by: `doc_extraction.py`**

These documents require robust text extraction and structural understanding.

-   **Source Ingestion**: The pipeline processes PDF and DOCX files from the `Insurance PDFs/` directory. For efficiency, only the **first 5 pages** of each PDF are processed.
-   **Extraction and Chunking**:
    -   **PDFs**: Text blocks and tables are extracted using PyMuPDF. Tables are chunked **row by row**, creating a separate document for each row to allow for fine-grained retrieval of specific details. Paragraphs are chunked by text block.
    -   **DOCX**: Documents are chunked **paragraph by paragraph** using python-docx library.
-   **Output**: Processed chunks are saved to `jsons_from_sources/all_pdf_chunks_cleaned.json` and `jsons_from_sources/all_docx_chunks_cleaned.json`

### 2. Angel One Support FAQs (Web-scraped)
**📁 Handled by: `web_scraper.py` → `angelone_faqs_postproc.py`**

This data is semi-structured and requires cleaning of web artifacts through a two-stage process:

#### Stage 1: Web Scraping (`web_scraper.py`)
-   **Dynamic Content Handling**: `Playwright` is used to handle dynamic, JavaScript-rendered content from Angel One support pages, ensuring all information is captured.
-   **HTML Parsing**: `BeautifulSoup` parses the final HTML and strips out boilerplate (navbars, footers, ads).
-   **Raw Data Storage**: Scraped content is stored in the `angelone_scraped_data/` directory for further processing.

#### Stage 2: FAQ Processing (`angelone_faqs_postproc.py`)
-   **Question-Answer Pair Extraction**: Identifies and extracts Q&A pairs from the scraped HTML. Each pair is treated as a single, coherent chunk to maintain the direct link between a question and its answer.
-   **Metadata Extraction**: Key metadata is attached to each chunk, including the original source `URL` for traceability and the `category/topic` (e.g., "Trading", "Account Opening") inferred from the page.
-   **HTML Tag Conversion**: Converts relevant HTML tags (`<ul>`, `<b>`, etc.) into clean, readable text.
-   **Output**: Processed FAQ chunks are saved to `jsons_from_sources/cleaned_angelone_faqs_chunks.json`

### 3. Final Cleaning and Filtering (Applied to All Sources)
**📁 Integrated within: `doc_extraction.py` and `angelone_faqs_postproc.py`**

After initial extraction and chunking, all chunks undergo a final, unified cleaning process:

-   **Boilerplate Removal**: Regex patterns remove common boilerplate content like page numbers (`Page 1 of 10`), copyright notices, and disclaimers.
-   **Short Chunk Removal**: Chunks with fewer than 10 words are discarded as they typically lack meaningful context for the LLM.
-   **Duplicate Removal**:
    -   **Exact duplicates** are removed.
    -   **Near-duplicates** are identified and removed using the Levenshtein distance similarity ratio. Chunks that are >95% similar are considered duplicates.

### 4. Vector Database Loading
**📁 Handled by: `setup_pinecone.py` → `pinecone_loader.py`**

The final step involves loading the processed data into Pinecone:

#### Setup (`setup_pinecone.py`)
-   **Index Creation**: Creates and configures the Pinecone vector database index with appropriate dimensions and similarity metrics.
-   **Environment Setup**: Handles API key configuration and connection establishment.

#### Data Loading (`pinecone_loader.py`)
-   **Embedding Generation**: Uses Sentence Transformers (all-MiniLM-L6-v2) to generate vector embeddings for each processed chunk.
-   **Namespace Organization**: Loads data into separate namespaces:
    -   `all-pdf-chunks-cleaned`: Health insurance PDF content
    -   `all-docx-chunks-cleaned`: Health insurance DOCX content  
    -   `cleaned-angelone-faqs-chunks`: Angel One FAQ content
-   **Batch Processing**: Efficiently uploads vectors in batches to Pinecone for optimal performance.

---

## 🔍 How It Works
**📁 Core System: `rag_system.py`**

1.  **Query Analysis**: Determines if the question is about health plans, Angel One, or irrelevant
2.  **Document Retrieval**: Searches relevant namespaces in Pinecone vector database
3.  **Reranking**: Uses LLM to rank documents by relevance to the specific query
4.  **Answer Generation**: Generates contextual responses using retrieved documents
5.  **Memory Management**: Maintains conversation history for context

## 💬 Example Queries

Try asking questions like:

-   **Health Plans**: "What are the eligibility criteria for health insurance?"
-   **Angel One**: "How do I apply for IPO in Angel One?"
-   **Irrelevant**: "What's the weather today?" (Will be handled appropriately)

## 🚨 Important Notes

-   **Data Privacy**: The use of open-source models like Mistral (as opposed to proprietary APIs like OpenAI) is a deliberate choice to enhance data privacy and control, with options for fully self-hosted, local deployments.
-   The system will respond appropriately to irrelevant queries.
-   All responses are generated based on the provided knowledge base.
-   The system maintains conversation history for better context understanding.
-   Agent logs are available in the UI for debugging and transparency.

## 🚀 Future Steps

-   **RAGAS Evaluation**: Implement the RAGAS framework (`eval_rag.py`) to rigorously evaluate and score the performance of the retrieval and generation pipeline, ensuring high-quality and factual responses.
-   **Multi-Language Support**: Extend the model's capabilities to understand and respond to queries in multiple languages to broaden accessibility.

## 🏗️ Local Development
**📁 Main Application: `streamlit_app.py`**

To run locally:

```bash
# Clone the repository
git clone <repo-url>
cd <repo-name>

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export PINECONE_API_KEY="your_key_here"
export HUGGING_FACE_HUB_TOKEN="your_token_here"

# Run the application
streamlit run streamlit_app.py
```

## 📁 File Structure Overview

```
├── streamlit_app.py              # Main Streamlit web application
├── rag_system.py                 # Multi-agent RAG system core
├── doc_extraction.py             # PDF/DOCX processing and chunking
├── web_scraper.py                # Angel One website scraping
├── angelone_faqs_postproc.py     # FAQ processing and cleaning
├── setup_pinecone.py             # Pinecone index setup
├── pinecone_loader.py            # Vector database loading
├── eval_rag.py                   # RAGAS evaluation framework
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker deployment configuration
├── .dockerignore                 # Docker build optimization
├── Insurance PDFs/               # Source health insurance documents
├── angelone_scraped_data/        # Raw scraped Angel One data
└── jsons_from_sources/           # Processed JSON chunks
    ├── all_pdf_chunks_cleaned.json
    ├── all_docx_chunks_cleaned.json
    └── cleaned_angelone_faqs_chunks.json
```

**Powered by LangChain, Hugging Face & Pinecone** 🚀
