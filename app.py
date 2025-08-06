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
import hashlib 

# Load environment variables from .env file
load_dotenv()

os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "Java TA Chatbot"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

st.set_page_config(
    page_title="Java TA Knowledge-Base Bot",
    page_icon="☕️",
    layout="wide",
)

st.markdown("""
<style>
  :root {
    --java-blue:   #5382a1;
    --java-orange: #e76f00;
    --java-cream:  #fdfaf6;
    --accent-green: #28a745; /* New: Green accent for success/buttons */
    --accent-red: #dc3545;   /* New: Red accent for warnings/errors */
    --light-blue-bubble: #E2ECFF; /* User bubble background */
    --dark-blue-text: #102A43;    /* User bubble text */
    --light-grey-bg: #F0F6FF;     /* Code block background */
    --dark-blue-code: #1E3A8A;    /* Code text */
  }
  body {
    background: linear-gradient(135deg, var(--java-blue) 0%, var(--java-orange) 120%);
    background-attachment: fixed;
  }
  /* make the white “card” readable */
  section.main > div:first-child {
    background: var(--java-cream) !important;
    border-radius: 1rem !important;
    padding: 2rem !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.08) !important;
  }

  /* chat bubbles override */
  div.stChatMessage.stChatMessage--user > div {
    background: var(--light-blue-bubble) !important;
    color: var(--dark-blue-text) !important;
    padding: 0.75rem 1rem !important;
    border-radius: 12px !important;
    margin: 0.5rem 0 !important;
    max-width: 70% !important;
    margin-left: auto !important;
  }
  div.stChatMessage.stChatMessage--assistant > div {
    background: #FFFFFF !important;
    color: #243B53 !important;
    padding: 0.75rem 1rem !important;
    border-radius: 12px !important;
    margin: 0.5rem 0 !important;
    max-width: 70% !important;
    margin-right: auto !important;
  }
  .stChatMessage__avatar { display: none !important; }

  /* code highlighting */
  code, pre {
    font-family: 'JetBrains Mono', monospace !important;
    background: var(--light-grey-bg) !important;
    color: var(--dark-blue-code) !important;
    border-radius: 8px !important;
  }
  pre {
    padding: 1rem !important;
    overflow-x: auto !important;
  }
  code {
    padding: 0.15rem 0.35rem !important;
  }

  /* file cards */
  .file-card {
    background: #FFFFFF;
    border: 1px solid rgba(0,0,0,0.06);
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    margin-bottom: 0.75rem;
    border-left: 5px solid var(--java-orange); /* New: Colored border */
  }

  div.stButton > button {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white !important;
  border-radius: 0.75rem;
  border: none;
  padding: 0.75rem 1.5rem;
  font-size: 1rem;
  font-weight: 500;
  transition: all 0.3s ease;
  box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
    }

    div.stButton > button:hover {
    background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%) !important;
    color: white !important;
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
    }

    div.stButton > button:active {
    background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%) !important;
    color: white !important;
    transform: translateY(0);
    box-shadow: 0 2px 10px rgba(102, 126, 234, 0.2);
    }

    div.stButton > button:focus {
    background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%) !important;
    color: white !important;
    outline: none;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.3);
    }

  /* Warnings */
  .stAlert.stAlert--warning {
      background-color: #fff3cd !important;
      color: #856404 !important;
      border-color: #ffeeba !important;
  }

  /* Info */
  .stAlert.stAlert--info {
      background-color: #d1ecf1 !important;
      color: #0c5460 !important;
      border-color: #bee5eb !important;
  }

  /* Success */
  .stAlert.stAlert--success {
      background-color: #d4edda !important;
      color: #155724 !important;
      border-color: #c3e6cb !important;
  }

</style>
""", unsafe_allow_html=True)

# Friendly hero banner  (main column, before chat)
st.markdown("""
<div style="margin-top:-20px">
<h1 style="font-size:2.3rem">🧑‍🏫 Java TA Chatbot</h1>
<p style="font-size:1.05rem">
Ask me anything about <strong>Java, OOP, UML, data structures, JVM internals</strong>—
Try e.g.:
<em>“Explain polymorphism in Java.”</em>
</p>
</div>
""", unsafe_allow_html=True)

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
        You may reformat, rename, and adapt examples from the context to answer the user’s question.
        Only if you’ve **tried both** factual lookup *and* generative synthesis (where allowed), **then** say:
            “Sorry, I couldn’t find that in the course material I was given.” and follow up with some counter questions related to the user question to make the user help you understand their question better.
        Do not include this apology if you’ve already answered the question or explained something from the context.
        📋 Answer Format:
        Brief Summary
        A one- or two-line direct answer to the question.
        Detailed Explanation
        A clear and structured explanation using the terminology and style of the UCL course.
        Java Code (if relevant)
        Provide working and formatted code blocks in:
        ```java
        // Code with meaningful comments
        public int square(int x) {
            'return x * x;'
        }
        ```
        Add comments or labels like // Constructor or // Method call example where helpful.
        Edge Cases & Pitfalls
        Briefly mention any exceptions, compiler warnings, gotchas, or common mistakes related to the topic.
        Optional Extras (only if helpful)
        ASCII-style diagrams for control flow, object relationships, or memory
        Small tables (e.g., lifecycle states, type conversions)

        🧩 📐 UML Diagrams (Usage Guidelines)
        When a question involves object-oriented design, class structure, inheritance, interfaces, or relationships between multiple classes, you may include a simple UML diagram to illustrate the structure.
        ✅ Use UML when:
        A student asks about class relationships (e.g., "How do these classes relate?")
        A concept involves inheritance, interfaces, composition, or abstract classes
        You are explaining object-oriented design patterns (e.g., Strategy, Factory, etc.)
        A student specifically asks you to create/draw a UML diagram
        ✅ Format:
        Use ASCII-style UML diagrams that clearly show class names, inheritance, fields, and methods
        Keep diagrams minimal and clean — no need to use full UML syntax or notation
        ✅ Examples:

        Inheritance Relationship:
        +----------------+
        |    Animal      |
        +----------------+
        | - name: String |
        +----------------+
        | +speak(): void |
        +----------------+
                ▲
                |
        +----------------+
        |     Dog        |
        +----------------+
        | +bark(): void  |
        +----------------+

        Interface Implementation:

        +--------------------+
        |   Flyable          |
        +--------------------+
        | +fly(): void       |
        +--------------------+

                ▲ implements
                |
        +----------------+
        |     Bird       |
        +----------------+
        | - wings: int   |
        | +fly(): void   |
        +----------------+

        Composition:

        +-------------------+
        |     House         |
        +-------------------+
        | - address: String |
        +-------------------+
        | +build(): void    |
        +-------------------+
                ◆
                |
        +-------------------+
        |     Room          |
        +-------------------+
        | - size: int       |
        +-------------------+

        Big UML Diagram Example:

                                ┌──────────────────────────┐
                                │        Employee          │
                                ├──────────────────────────┤
                                │ - name        : String   │
                                │ - department  : String   │
                                │ - monthlyPay  : int      │
                                ├──────────────────────────┤
                                │ +String getName()        │
                                │ +String getDepartment()  │
                                │ +int    getMonthlyPay()  │
                                └──────────────────────────┘
                                        ▲
                        ┌──────────────────┴──────────────────┐
                        │                                     │
            ┌──────────────────────────┐        ┌──────────────────────────┐
            │         Manager          │        │          Worker          │
            ├──────────────────────────┤        ├──────────────────────────┤
            │ - bonus        : int     │        │ (no extra fields)        │
            ├──────────────────────────┤        ├──────────────────────────┤
            │ +int getMonthlyPay()     │        │                          │
            └──────────────────────────┘        └──────────────────────────┘
                        ▲ 0..* (managed by ExecutiveTeam)
                        │
                        │
                        │               1
                ┌──────────────────────────┴──────────────────────────┐
                │                   ExecutiveTeam                     │
                ├──────────────────────────────────────────────────────┤
                │ +void add(Manager manager)                          │
                │ +void remove(String name)                           │
                └──────────────────────────────────────────────────────┘
                        ▲ 1 (created/owned by Company)
                        │
                        │
                        │
        ┌───────────────────────────────────────────────────────────────┐
        │                           Company                            │
        ├───────────────────────────────────────────────────────────────┤
        │ - name : String                                               │
        ├───────────────────────────────────────────────────────────────┤
        │ +void addWorker(String name, String department, int pay)      │
        │ +void addManager(String name, String department, int pay,     │
        │                      int bonus)                               │
        │ +void addToExecutiveTeam(Manager manager)                     │
        │ +int  getTotalPayPerMonth()                                   │
        └───────────────────────────────────────────────────────────────┘
                        | 1
                        | has
                        | 0..*
                        ▼
                ┌──────────────────────────┐
                │        Employee          │  (same box as above; association shown here)
                └──────────────────────────┘

        ✅ Explain the diagram in words:
        “In this example, Dog inherits from Animal. The base class provides the speak() method, and Dog adds a new method bark().”
        ❌ Don’t use UML for simple method questions or unrelated procedural logic.

        Mini Quiz (optional)
        Occasionally include a short quiz question to reinforce learning (e.g., “What would happen if the return type was void?”). Include answers at the end.
        ✏️ Formatting Rules:
        Use correct Java identifier formatting (e.g., MyClass, toString(), ArrayList<Integer>)
        Use bullet points or subheadings where clarity improves
        Do not include material or Java APIs not explicitly referenced in the context
        ⚠️ Handling Common Cases:
        If the user question is too vague, explain a general case using course-relevant examples (e.g., square(int x) or sayHello()).
        If multiple interpretations of a question are possible, briefly list the plausible ones and address each.
        If the question mentions a Java keyword (e.g., final, static, record), define it precisely and relate it to context.
        If the question is about bugs, compilation errors, or design, point to patterns, methods, or design tips from the context material.
        🎓 Teaching Style:
        Be professional, supportive, and clear — like a trusted lab demonstrator or tutor.
        Prioritize conceptual clarity over fancy language.
        Avoid filler. Never speculate.
        Structure your answer to help students understand, not just memorize.
        🧠 Self-Check Before Answering:
        Ask yourself: 1. "If it is a UML diagram, use examples in your prompt and answer."
                    2. “Else, can I find any relevant example, definition, or code in the context or the prompt that helps answer this question?”
        If yes, adapt and use it.
        If no, say: “Sorry, I couldn’t find that in the course material I was given.” and follow up with some counter questions related to the user question to make the user help you understand their question better.

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

# uploaded_file = st.file_uploader("📎 Attach a PDF", type=["pdf"])
with st.sidebar:
    st.header("📎 Upload a PDF")
    uploaded_file = st.file_uploader("Choose a Java/OOP PDF", type=["pdf"])
    st.markdown("---")
    st.caption("🔹 Diagrams & tables auto-extracted 🔹")
    st.markdown("---")
    st.header("⚡ Quick Actions")
    if st.button("Explain Inheritance"):
        st.session_state.quick = "Explain inheritance in Java"
    if st.button("Show UML Example"):
        st.session_state.quick = "Give me a UML class diagram example in Java"
    if st.button("List OOP Principles"):
        st.session_state.quick = "What are the four main OOP principles in Java?"

if "quick" in st.session_state:
    # use the quick action text and then clear it
    user_input = st.session_state.pop("quick")
else:
    user_input = st.chat_input("Ask me anything…")


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
# ------------------------------------------------------------------
# ⇨ MOD ❼  Processed-files section: use tabs & card style
# ------------------------------------------------------------------
files_tab, about_tab = st.tabs(["📁 Processed Files", "ℹ️ About"])

if st.session_state.get("show_processed_files", False):
    with files_tab:
        if st.session_state.processed_files:
            for i, file_info in enumerate(st.session_state.processed_files):
                with st.container():
                    st.markdown(f"<div class='file-card'>"
                                f"<strong>{file_info['name']}</strong> "
                                f"({file_info['size']:,} bytes)"
                                f"<br><code>{file_info['hash'][:16]}…</code>"
                                "</div>", unsafe_allow_html=True)
        else:
            st.info("No files processed yet.")
    
    if st.button("✖️ Hide processed files"):
        st.session_state.show_processed_files = False
        st.rerun()

with about_tab:
    st.markdown("""
    **Java TA Bot** leverages LangChain, GPT-4o, and ChromaDB to provide
    accurate answers based on your course PDFs.
    """)

# Render the full chat history (user + assistant messages)
# ------------------------------------------------------------------
# ⇨ MOD ❻  Chat message rendering uses CSS classes
# ------------------------------------------------------------------
for msg in st.session_state.chat_history:
    role_class = "user-msg" if msg["role"] == "user" else "bot-msg"
    with st.chat_message(msg["role"]):
        st.markdown(f"<div class='{role_class}'>{msg['content']}</div>",
                    unsafe_allow_html=True)

# Chat-style input field
# user_input = st.chat_input("Ask me anything...")

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