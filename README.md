# Java TA Chatbot

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)](https://streamlit.io/)

A Retrieval-Augmented Generation (RAG) chatbot designed to help students study Java and Object-Oriented Programming (OOP) topics. The application provides intelligent document processing, visual content analysis, and contextual question-answering using course PDFs and generated Q&A pairs.

## 🎥 Video Demo Link: 

https://youtu.be/cDILKAl2BuU

## 🚀 Features

- **📚 RAG-powered Knowledge Base**: Index PDFs and generated Q&A into a persistent ChromaDB vector store
- **💬 Interactive Streamlit UI**: Clean, modern chat interface with quick action buttons and styled chat bubbles
- **📄 Advanced PDF Processing**: 
  - Extracts text and structured tables
  - Detects and analyzes visual elements (diagrams, tables, images)
  - Uses GPT-4 Vision to describe visual content
  - Smart content chunking for optimal retrieval
- **🧠 Conversation Memory**: Maintains chat history and context through LangChain's ConversationBufferMemory
- **🔄 Session-based Document Upload**: Upload and process PDFs during chat sessions with deduplication
- **🔍 Vector Search**: Semantic search over embedded documents using OpenAI embeddings
- **📊 LangSmith Tracing**: Trace RAG calls for debugging and evaluation

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Preparing the Knowledge Base](#preparing-the-knowledge-base)
  - [Running the Application](#running-the-application)
- [Architecture](#architecture)
- [Troubleshooting](#troubleshooting)
- [Security](#security)

## Prerequisites

- **Python**: 3.10 or higher (3.10+ recommended)
- **OpenAI API Key**: Required for embeddings and LLM responses
- **LangSmith Account** (Optional): For tracing and monitoring RAG calls

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Java-TA-Chatbot
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
```

**Activate the virtual environment:**

- On macOS/Linux:
  ```bash
  source .venv/bin/activate
  ```

- On Windows:
  ```bash
  .venv\Scripts\activate
  ```

### 3. Install Dependencies

```bash
python3 -m pip install --upgrade pip
python3 -m pip install pymupdf openai chromadb tqdm nltk tiktoken python-docx langchain langsmith langchain_openai moviepy streamlit python-dotenv pillow
```

**Note:** If you encounter SSL errors on macOS (common with NLTK), the notebook includes SSL workarounds and will download required NLTK data automatically.

## Configuration

### Environment Variables

Create a `.env` file in the project root directory:

```bash
# Required: OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Optional: LangSmith/LangChain Tracing
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_PROJECT=Java TA Chatbot
LANGCHAIN_TRACING_V2=true
```

**Important Notes:**
- The application uses `python-dotenv` to load environment variables via `load_dotenv()`
- Never commit your `.env` file or API keys to version control
- LangSmith variables are optional and can be omitted if you don't need tracing

## Project Structure

```
Java-TA-Chatbot/
├── app.py                 # Main Streamlit application (chat interface + PDF processing)
├── frontend.py            # UI styling and sidebar components
├── helper.py              # PDF processing utilities (visual detection, chunking)
├── rag_pipeline.py        # RAG pipeline implementation (embedding, retrieval, generation)
├── java_ta.ipynb          # Jupyter notebook for knowledge base preparation
├── knowledge_base_data/   # Directory for course materials
│   ├── course_info/       # Course information PDFs
│   ├── exercises/         # Exercise sheets and solutions
│   ├── lecture_notes/     # Lecture slides and notes
│   ├── past_papers/       # Past exam papers
│   └── videos/            # Video materials (if any)
├── chroma_db/             # Persistent ChromaDB vector store (auto-created)
├── .env                   # Environment variables (not committed)
└── README.md             # This file
```

## Usage

### Preparing the Knowledge Base

The knowledge base must be populated before using the chat interface. Use the `java_ta.ipynb` notebook to ingest documents into ChromaDB.

#### Option A: Exam-style Q&A Extraction and Generation

1. **Place PDFs**: Copy past exam papers into `knowledge_base_data/past_papers/`

2. **Run Notebook Cells**: Execute the relevant cells in `java_ta.ipynb` to:
   - Extract exam-style questions from PDFs (detects pages with marks or numbered lists)
   - Optionally attach page images for visual context
   - Generate clear answers using GPT-4o
   - Chunk question/answer text using an overlap-aware sentence tokenizer
   - Embed chunks with `text-embedding-3-small`
   - Store in the persistent `knowledge-base6` collection in ChromaDB

   **Relevant cells** (may vary):
   - Text/image extraction, question parsing: Cells ~10-16
   - Answer generation and embedding: Cells ~14, 20

#### Option B: Generic PDF Chunking and Embedding

1. **Place PDFs**: Copy PDFs to index in your working directory (or adjust paths in the notebook)

2. **Run Generic Ingestion Cells**: Execute cells around Cell 22 to:
   - Extract raw text from PDFs
   - Chunk text intelligently
   - Embed with `text-embedding-3-small`
   - Store in `knowledge-base6`

**Note:** ChromaDB storage persists under `./chroma_db` and can be reused across application runs.

### Running the Application

Start the Streamlit application from the project root:

```bash
streamlit run app.py
```

The application will automatically open in your default web browser, typically at `http://localhost:8501`.

#### UI Guide

- **Sidebar**:
  - Upload PDF files for session-based processing
  - Quick action buttons for common Java/OOP queries:
    - "Explain Inheritance"
    - "Show UML Example"
    - "List OOP Principles"

- **Main Chat Interface**:
  - Enter questions in the chat input at the bottom
  - The application retrieves relevant chunks from the knowledge base
  - Responses include contextual information from course materials

- **PDF Processing**:
  - Uploaded PDFs are processed with:
    - Text extraction
    - Table detection and extraction
    - Visual element analysis (diagrams, images)
    - Smart chunking for optimal context retrieval
  - Processed content is available as additional context for the current session
  - Duplicate uploads are detected and prevented

#### RAG Flow

The application follows this flow for each query:

1. **Embedding**: Create embeddings for the user query using `text-embedding-3-small`
2. **Retrieval**: Retrieve top-k (default: 5) relevant chunks from ChromaDB
3. **Context Assembly**: Build a prompt including:
   - Retrieved chunks from the knowledge base
   - Conversation history from memory
   - Optional: Session-uploaded PDF content
4. **Generation**: Generate answer using GPT-4o with context-aware prompting

## Architecture

### Components

- **UI Layer** (`app.py`, `frontend.py`):
  - Streamlit-based web interface
  - File upload handling
  - Chat history management
  - Session state management

- **Processing Layer** (`helper.py`):
  - PDF parsing using PyMuPDF (fitz)
  - Visual element detection (tables, images, drawings)
  - GPT-4 Vision integration for visual analysis
  - Smart content chunking

- **RAG Pipeline** (`rag_pipeline.py`):
  - Query embedding generation
  - Vector similarity search in ChromaDB
  - Prompt construction with context
  - LLM response generation

- **Vector Database**:
  - ChromaDB persistent client at `./chroma_db`
  - Collection: `knowledge-base6`
  - Uses OpenAI `text-embedding-3-small` for embeddings

### Models

- **Embeddings**: `text-embedding-3-small` (OpenAI)
- **Text Generation**: `gpt-4o` (OpenAI ChatOpenAI)
- **Vision Analysis**: `gpt-4o` with vision capabilities

### Memory Management

- Uses LangChain's `ConversationBufferMemory` for conversation history
- Stores chat history in Streamlit session state
- Maintains context across conversation turns

## Troubleshooting

### Common Issues

**"OpenAI key not found"**
- Ensure `.env` file exists in the project root
- Verify `OPENAI_API_KEY` is set correctly
- Restart your shell or IDE after creating `.env`

**LangSmith not collecting traces**
- Set `LANGCHAIN_API_KEY` in `.env`
- Enable `LANGCHAIN_TRACING_V2=true`
- Verify LangSmith credentials are valid

**ChromaDB permission errors**
- Ensure `./chroma_db` directory is writable
- Check file system permissions
- Create the directory manually if it doesn't exist: `mkdir chroma_db`

**NLTK errors (macOS SSL)**
- The notebook includes SSL workarounds
- NLTK will automatically download required data (`punkt`, `punkt_tab`)
- If issues persist, manually download NLTK data:
  ```python
  import nltk
  nltk.download('punkt')
  nltk.download('punkt_tab')
  ```

**Empty retrievals**
- Verify documents were embedded successfully
- Check that documents exist in the `knowledge-base6` collection
- Re-run the notebook ingestion process if needed

**PDF processing errors**
- Ensure PDFs are not corrupted or password-protected
- Check that PyMuPDF (pymupdf) is installed correctly
- Verify sufficient disk space for temporary file processing

## Security

### Data Privacy

- **API Keys**: Never commit `.env` files or API keys to version control
- **Local Storage**: Vector database is stored locally by default (`./chroma_db`)
- **Session Data**: Chat history is stored in Streamlit session state (cleared on refresh)

### Best Practices

- Use environment variables for all sensitive information
- Regularly back up your `chroma_db` directory if it contains valuable data
- Review uploaded PDFs before processing (ensure no sensitive data)
- Keep dependencies up to date for security patches

### Backing Up Data

To back up your knowledge base:

```bash
# Copy the entire chroma_db directory
cp -r chroma_db chroma_db_backup

# Or create a zip archive
zip -r chroma_db_backup.zip chroma_db/
```

## Acknowledgments

This project is built with:

- [Streamlit](https://streamlit.io/) - Web application framework
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [LangChain](https://www.langchain.com/) - LLM application framework
- [LangSmith](https://smith.langchain.com/) - LLM observability platform
- [OpenAI](https://openai.com/) - Embedding and language models
- [PyMuPDF](https://pymupdf.readthedocs.io/) - PDF processing library

---

**Note:** This chatbot is designed specifically for Java and OOP course materials. Adapt the prompts and processing logic in `rag_pipeline.py` and `helper.py` if you want to use it for other subjects.
