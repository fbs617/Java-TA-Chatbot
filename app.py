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

# Load environment variables from .env file
load_dotenv()

os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "Java TA Chatbot"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# ------------------------------------------------------------------------Helper functions-----------------------------------------------------------------------------------
def extract_pages(pdf_path):
    """Extracts text + images from each page"""
    doc = fitz.open(pdf_path)
    page_data = []

    for i, page in enumerate(doc):
        text = page.get_text().strip()
        pix = page.get_pixmap(dpi=100)
        img_bytes = pix.tobytes("png")
        base64_img = base64.b64encode(img_bytes).decode("utf-8")
        page_data.append({"page": i + 1, "text": text, "image": base64_img})
    return page_data
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Initialize LangChain memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# Initialize Streamlit frontend chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Holds the concatenated plain-text of all PDFs the user has attached
if "session_docs" not in st.session_state:
    st.session_state.session_docs = ""


@traceable(name="RAG_Chatbot_Answer")
def rag_answer2(query, collection, memory, attachment_text="", embedding_model="text-embedding-3-small", k=3):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # keep for embeddings

    # Step 1: Embed the user's question
    query_embedding = client.embeddings.create(
        input=query,
        model=embedding_model
    ).data[0].embedding

    # Step 2: Retrieve top-k chunks from Chroma
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    relevant_chunks = results["documents"][0]

     # Get memory history as string
    memory_context = memory.buffer

    # Step 3: Construct prompt
    context = "\n\n".join(relevant_chunks)

    if attachment_text:           
        context += (
            "\n\n────────  📄 User Attachment  ────────\n\n"
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

    # Step 4: LangSmith-traceable LLM call
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
if uploaded_file:
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    pages = extract_pages(tmp_path)          # ← your helper
    # --- NEW: build Markdown that interleaves text + the page image ----
    pdf_blocks = []
    for pg in pages:
        txt = pg["text"].strip()
        # img_md = f'![page-{pg["page"]}](data:image/png;base64,{pg["image"]})'
        img_md = []
        # keep both even if one is empty
        pdf_blocks.append("\n\n".join(part for part in (txt, img_md) if part))

    pdf_md = "\n\n".join(pdf_blocks)

    # Store for the rest of the chat
    st.session_state.session_docs += "\n\n" + pdf_md

    st.success("PDF attached – text **and images** added to context ✅")

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