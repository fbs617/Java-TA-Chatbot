"""
This script implements a Streamlit application that serves as a chatbot for a Java TA (Teaching Assistant) system. 
It allows users to upload PDF files, processes the content for visual analysis, and maintains a chat history 
for user interactions. The application utilizes various libraries such as LangChain, ChromaDB, and OpenAI to 
facilitate conversation memory, document processing, and retrieval-augmented generation (RAG) for responses.
"""

import streamlit as st
import chromadb
from langchain_openai import ChatOpenAI
from langsmith import traceable
from openai import OpenAI
import os
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
import tempfile      # Save uploads for PyMuPDF
import hashlib 

# imports from other files
from frontend import styles, sidebar
from helper import smart_chunk_content, extract_pages_optimized
from rag_pipeline import rag_pipeline

# Load environment variables
load_dotenv()

os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "Java TA Chatbot"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Streamlit frontend setup
styles()

# Sidebar for file upload and quick actions
uploaded_file = sidebar()

# Initialize conversation memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# Initialize chat history (UI)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Store processed PDF content
if "session_docs" not in st.session_state:
    st.session_state.session_docs = ""

# Store processed file metadata
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# Initialize ChromaDB and collection
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("knowledge-base6")

if "quick" in st.session_state:
    # Use the quick action text, then clear it
    user_input = st.session_state.pop("quick")
else:
    user_input = st.chat_input("Ask me anything…")

# Only process file if it is uploaded
if uploaded_file is not None:
    # Compute file hash to detect duplicates
    file_content = uploaded_file.read()
    file_hash = hashlib.sha256(file_content).hexdigest()
    uploaded_file.seek(0)  # Reset file pointer
    
    # Track the current upload with a session key
    current_file_key = f"{uploaded_file.name}_{file_hash}"
    
    # Only warn or process on a new upload
    if "last_processed_file" not in st.session_state:
        st.session_state.last_processed_file = None
    
    # Skip re-processing the same upload on refresh
    if st.session_state.last_processed_file != current_file_key:
        # Check if already processed this session
        already_processed = any(f["hash"] == file_hash for f in st.session_state.processed_files)
        
        if not already_processed:
            # Add to processed files
            st.session_state.processed_files.append({
                "name": uploaded_file.name,
                "hash": file_hash,
                "size": uploaded_file.size
            })
            
            # Update the last processed file marker
            st.session_state.last_processed_file = current_file_key
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
                
            with st.spinner("🔍 Processing PDF with smart visual analysis..."):
                # Run optimized extraction
                pages = extract_pages_optimized(tmp_path)
                
                # Chunk content for retrieval
                chunks = smart_chunk_content(pages)
                
                # Combine chunk text for this session
                all_content = []
                for chunk in chunks:
                    content_with_meta = f"[Page {chunk['metadata']['page']}, Chunk {chunk['metadata']['chunk']}]\n{chunk['text']}"
                    all_content.append(content_with_meta)
                
                combined_content = "\n\n".join(all_content)
                
                st.session_state.session_docs += "\n\n" + combined_content
                
                # Show processing stats
                total_pages = len(pages)
                pages_with_visuals = sum(1 for p in pages if p["has_visuals"])
                
                st.success(f"""PDF processed successfully! ✅
                
        **Processing Summary:**
        - 📄 **{total_pages} pages** processed
        - 🎨 **{pages_with_visuals} pages** contained visual elements (diagrams, tables, etc.)
        - 📊 **{len(chunks)} content chunks** created for optimal context retrieval
                
        Visual content has been analyzed and converted to text descriptions that preserve the meaning of diagrams, tables, and other non-text elements.""")
        
        else:
            # Warn only for new duplicate uploads
            st.session_state.last_processed_file = current_file_key
            st.warning(f"⚠️ File '{uploaded_file.name}' has already been processed in this session!")
            
            # Provide a reprocess option
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Process anyway", key="reprocess_btn"):
                    # Remove from processed list and reset
                    st.session_state.processed_files = [
                        f for f in st.session_state.processed_files 
                        if f["hash"] != file_hash
                    ]
                    st.session_state.last_processed_file = None
                    st.rerun()
            
            with col2:
                if st.button("📋 Show processed files", key="show_files_btn"):
                    st.session_state.show_processed_files = True

# Render chat history (user and assistant)
# Chat message rendering uses CSS classes
for msg in st.session_state.chat_history:
    role_class = "user-msg" if msg["role"] == "user" else "bot-msg"
    with st.chat_message(msg["role"]):
        st.markdown(f"<div class='{role_class}'>{msg['content']}</div>",
                    unsafe_allow_html=True)

# Chat input
# user_input = st.chat_input("Ask me anything...")

# Handle the user's question
if user_input:
    # Add user message to memory and UI
    st.session_state.memory.chat_memory.add_user_message(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Call the RAG function
    with st.spinner("Thinking..."):
        response = rag_pipeline(user_input, collection, memory=st.session_state.memory, attachment_text=st.session_state.session_docs)

    # Add assistant response to memory and UI
    st.session_state.memory.chat_memory.add_ai_message(response)
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    st.rerun()  # Rerun to display the new messages