"""
This script contains the Streamlit application for a Java TA Knowledge-Base Chatbot. 
It provides functionalities to set the page styles, layout, and sidebar components for user interaction. 
The `styles` function configures the visual appearance of the app, including custom CSS for chat bubbles, 
buttons, alerts, and overall layout. The `sidebar` function allows users to upload PDF documents related to 
Java and OOP, and offers quick action buttons for common queries about Java concepts.
"""

import streamlit as st

def styles():

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

    # Hero banner (main column, before chat)
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

def sidebar():
    
    """
    Sidebar function for a Streamlit application that allows users to upload a PDF file 
    related to Java or Object-Oriented Programming (OOP). It provides quick action buttons 
    to explain inheritance, show a UML example, and list OOP principles. The uploaded file 
    is returned for further processing.
    Returns:
        uploaded_file: The PDF file uploaded by the user.
    """

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
    return uploaded_file