import streamlit as st
import chromadb
from langchain_openai import ChatOpenAI
from langsmith import traceable
from openai import OpenAI
import os
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
import fitz          # PyMuPDF  – PDF parsing
import base64        # encode images
import tempfile      # save upload to disk for PyMuPDF
from PIL import Image
import io
import hashlib  # Add this import at the top

# Load environment variables from .env file
load_dotenv()

os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "Java TA Chatbot"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# ------------------------------------------------------------------------Helper functions-----------------------------------------------------------------------------------

def detect_visual_elements(page):
    """Detect if a page contains tables, diagrams, or other visual elements"""
    # Check for tables
    table_finder = page.find_tables()
    tables = list(table_finder)  # Convert TableFinder to list
    
    # Check for images
    images = page.get_images()
    
    # Check for drawings (vector graphics)
    drawings = page.get_drawings()
    # Simple heuristic: if text is sparse but there are visual elements, it's likely a diagram
    text = page.get_text().strip()
    text_density = len(text) / (page.rect.width * page.rect.height) if page.rect.width * page.rect.height > 0 else 0
    
    has_visual_content = len(tables) > 0 or len(images) > 0 or len(drawings) > 5 or text_density < 0.001
    
    return has_visual_content, len(tables), len(images), len(drawings)

def extract_table_text(page):
    """Extract tables as structured text"""
    table_finder = page.find_tables()
    tables = list(table_finder)  # Convert TableFinder to list
    table_texts = []
    
    for table in tables:
        try:
            table_data = table.extract()
            if table_data:
                # Convert table to markdown format
                markdown_table = "| " + " | ".join(str(cell) if cell else "" for cell in table_data[0]) + " |\n"
                markdown_table += "|" + "---|" * len(table_data[0]) + "\n"
                
                for row in table_data[1:]:
                    markdown_table += "| " + " | ".join(str(cell) if cell else "" for cell in row) + " |\n"
                
                table_texts.append(f"TABLE:\n{markdown_table}\n")
        except:
            continue
    
    return "\n".join(table_texts)

def compress_image(pix, max_size=(800, 600), quality=85):
    """Compress image to reduce token usage"""
    img_bytes = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_bytes))
    
    # Resize if too large
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    # Convert to RGB if necessary and save as JPEG for better compression
    if img.mode in ('RGBA', 'LA', 'P'):
        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'P':
            img = img.convert('RGBA')
        background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
        img = background
    
    output = io.BytesIO()
    img.save(output, format='JPEG', quality=quality, optimize=True)
    return base64.b64encode(output.getvalue()).decode("utf-8")

def analyze_visual_content_with_gpt4v(base64_image, client):
    """Use GPT-4V to analyze visual content and convert to text description"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Analyze this image from a Java programming course document. Describe any:
1. UML diagrams (class diagrams, sequence diagrams, etc.) - describe the classes, relationships, methods, and fields
2. Code snippets or examples - transcribe the code exactly
3. Flowcharts or algorithmic diagrams - describe the logic flow
4. Tables with data - convert to markdown table format
5. Mathematical formulas or expressions
6. Any other educational content relevant to Java programming

Be precise and detailed in your description so it can be used as context for answering student questions."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Error analyzing visual content: {str(e)}]"

def extract_pages_optimized(pdf_path):
    """Extracts text + analyzes visual content efficiently"""
    doc = fitz.open(pdf_path)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    page_data = []

    for i, page in enumerate(doc):
        text = page.get_text().strip()
        
        # Extract tables as structured text
        table_text = extract_table_text(page)
        
        # Check if page has significant visual content
        has_visual, num_tables, num_images, num_drawings = detect_visual_elements(page)
        
        visual_description = ""
        if has_visual and (num_images > 0 or num_drawings > 10 or (len(text) < 100 and num_drawings > 0)):
            # Only process visual content for pages that likely contain diagrams/important visuals
            pix = page.get_pixmap(dpi=150)  # Higher DPI for better OCR
            compressed_img = compress_image(pix)
            visual_description = analyze_visual_content_with_gpt4v(compressed_img, client)
            
            st.info(f"📊 Analyzed visual content on page {i+1}: {num_tables} tables, {num_images} images, {num_drawings} drawings")
        
        # Combine all content
        combined_content = []
        if text:
            combined_content.append(f"TEXT CONTENT:\n{text}")
        if table_text:
            combined_content.append(f"STRUCTURED TABLES:\n{table_text}")
        if visual_description:
            combined_content.append(f"VISUAL ELEMENTS DESCRIPTION:\n{visual_description}")
        
        page_content = "\n\n".join(combined_content)
        page_data.append({
            "page": i + 1, 
            "content": page_content,
            "has_visuals": has_visual,
            "stats": f"Tables: {num_tables}, Images: {num_images}, Drawings: {num_drawings}"
        })
    
    return page_data

# Alternative: Smart chunking approach
def smart_chunk_content(page_data, max_chunk_size=4000):
    """Break content into smart chunks that respect logical boundaries"""
    chunks = []
    
    for page in page_data:
        content = page["content"]
        page_num = page["page"]
        
        if len(content) <= max_chunk_size:
            chunks.append({
                "text": content,
                "metadata": {"page": page_num, "chunk": 1}
            })
        else:
            # Split by sections, preserving visual elements together
            sections = content.split("\n\n")
            current_chunk = ""
            chunk_num = 1
            
            for section in sections:
                if len(current_chunk) + len(section) + 2 <= max_chunk_size:
                    current_chunk += section + "\n\n"
                else:
                    if current_chunk:
                        chunks.append({
                            "text": current_chunk.strip(),
                            "metadata": {"page": page_num, "chunk": chunk_num}
                        })
                        chunk_num += 1
                    current_chunk = section + "\n\n"
            
            if current_chunk:
                chunks.append({
                    "text": current_chunk.strip(),
                    "metadata": {"page": page_num, "chunk": chunk_num}
                })
    
    return chunks

# ------------------------------------------------------------------------End Helper functions------------------------------------------------------------------------

# Initialize LangChain memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# Initialize Streamlit frontend chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Holds the processed content from PDFs
if "session_docs" not in st.session_state:
    st.session_state.session_docs = ""

# Stores metadata of processed files
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

@traceable(name="RAG_Chatbot_Answer")
def rag_answer2(query, collection, memory, attachment_text="", embedding_model="text-embedding-3-small", k=5):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Step 1: Embed the user's question
    query_embedding = client.embeddings.create(
        input=query,
        model=embedding_model
    ).data[0].embedding

    # Step 2: Retrieve top-k chunks from Chroma (increased k since we're chunking better)
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    relevant_chunks = results["documents"][0]

    # Get memory history as string
    memory_context = memory.buffer

    # Step 3: Construct prompt
    context = "\n\n".join(relevant_chunks)

    if attachment_text:           
        context += (
            "\n\n────────  📄 User Attachment (Current Session)  ────────\n\n"
            + attachment_text.strip()
        )

    # [Rest of your rag_answer2 function remains the same...]
    full_prompt = f"""
    Previous Conversation:
        {memory_context}
        You are a helpful and expert Java teaching assistant at UCL. You assist students by answering their questions using only the course material provided in the context.
        Your answers must always be:
        Accurate, based solely on the context below;
        Thorough, with clear explanations and examples when relevant;
        Friendly and pedagogical, like a knowledgeable TA during office hours.
        🔍 Context Usage Instructions:
        If the user asks you to generate **new teaching materials** (exam papers, quizzes, exercises, sample projects), you should **synthesize** them using the topics, code examples, and explanations from the context—even if no exact exam exists there.
        If the user explicitly asks you to draw or create a UML diagram, you may rely on the UML Diagrams (Usage Guidelines) section in this prompt—even though no UML lives in the context.
        Otherwise, use only the information found in the context. Do not invent APIs, methods, definitions, or facts.
        You may reformat, rename, and adapt examples from the context to answer the user's question.
        Only if you've **tried both** factual lookup *and* generative synthesis (where allowed), **then** say:
            "Sorry, I couldn't find that in the course material I was given." and follow up with some counter questions related to the user question to make the user help you understand their question better.
        Do not include this apology if you've already answered the question or explained something from the context.

        Context:
        {context}

        Question:
        {query}

        Answer:
        """

    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    response = llm.invoke(full_prompt)

    return response.content

# Initialize ChromaDB client and collection
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("knowledge-base6")

# Streamlit app layout
st.title("💬 Knowledge Base Chatbot")
st.markdown("Ask anything from the knowledge base below.")

uploaded_file = st.file_uploader("📎 Attach a PDF", type=["pdf"])

# Only process when a file is actually uploaded
if uploaded_file is not None:
    # Calculate file hash for duplicate detection
    file_content = uploaded_file.read()
    file_hash = hashlib.sha256(file_content).hexdigest()
    uploaded_file.seek(0)  # Reset file pointer
    
    # Check if this specific file upload is new (using a session state key)
    current_file_key = f"{uploaded_file.name}_{file_hash}"
    
    # Only show duplicate warning or process if this is a new file interaction
    if "last_processed_file" not in st.session_state:
        st.session_state.last_processed_file = None
    
    # Check if this is the same file upload as before (to prevent re-processing on page refresh)
    if st.session_state.last_processed_file != current_file_key:
        # Check if already processed in this session
        already_processed = any(f["hash"] == file_hash for f in st.session_state.processed_files)
        
        if not already_processed:
            # Add to processed files list
            st.session_state.processed_files.append({
                "name": uploaded_file.name,
                "hash": file_hash,
                "size": uploaded_file.size
            })
            
            # Update the last processed file
            st.session_state.last_processed_file = current_file_key
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
                
            with st.spinner("🔍 Processing PDF with smart visual analysis..."):
                # Use the optimized extraction
                pages = extract_pages_optimized(tmp_path)
                
                # Create chunks for better context management
                chunks = smart_chunk_content(pages)
                
                # Combine all content for the session
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
            # Only show this warning for new duplicate uploads
            st.session_state.last_processed_file = current_file_key
            st.warning(f"⚠️ File '{uploaded_file.name}' has already been processed in this session!")
            
            # Optional: Show reprocess button
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

# Optional: Show processed files section
if st.session_state.get("show_processed_files", False):
    st.subheader("📁 Processed Files in This Session")
    
    if st.session_state.processed_files:
        for i, file_info in enumerate(st.session_state.processed_files):
            with st.expander(f"📄 {file_info['name']} ({file_info['size']:,} bytes)"):
                st.write(f"**File Hash:** `{file_info['hash'][:16]}...`")
                
                if st.button(f"🗑️ Remove from session", key=f"remove_{i}"):
                    st.session_state.processed_files.pop(i)
                    st.rerun()
    else:
        st.info("No files processed yet.")
    
    if st.button("✖️ Hide processed files"):
        st.session_state.show_processed_files = False
        st.rerun()

# Render the full chat history (user + assistant messages)
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat-style input field
user_input = st.chat_input("Ask me anything...")

# Process the user's question
if user_input:
    # Add user message to both LangChain memory and UI chat history
    st.session_state.memory.chat_memory.add_user_message(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Call your RAG function
    with st.spinner("Thinking..."):
        response = rag_answer2(user_input, collection, memory=st.session_state.memory, attachment_text=st.session_state.session_docs)

    # Add assistant message to memory and UI
    st.session_state.memory.chat_memory.add_ai_message(response)
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    st.rerun()  # rerun to show the new messages