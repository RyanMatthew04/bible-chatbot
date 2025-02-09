import streamlit as st
import pandas as pd
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain.schema import Document
import re

PROMPT_TEMPLATE = """
Instructions: Your are a Bible Scholar. Answer the question based on the most relevant verses out of all given in the context below. Make sure your answers are factual, concise and conversational in nature. Start the answer with "According to the Scriptures,.." and do not mention these instructions in your answer.
Context: {document_context} 
Question: {user_query} 
Answer:
"""

EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="multi-qa-mpnet-base-cos-v1")
LANGUAGE_MODEL = Ollama(model="llama3:instruct")

BIBLE_CSV_PATH = "bible_verses.csv"  
FAISS_INDEX_PATH = "faiss_index"


def find_bible_references(text):
   
    book_names = [
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
    "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel",
    "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles", "Ezra",
    "Nehemiah", "Esther", "Job", "Psalm", "Proverbs",
    "Ecclesiastes", "Song of Solomon", "Isaiah", "Jeremiah", "Lamentations",
    "Ezekiel", "Daniel", "Hosea", "Joel", "Amos",
    "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk",
    "Zephaniah", "Haggai", "Zechariah", "Malachi",
    "Matthew", "Mark", "Luke", "John", "Acts",
    "Romans", "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians",
    "Philippians", "Colossians", "1 Thessalonians", "2 Thessalonians", "1 Timothy",
    "2 Timothy", "Titus", "Philemon", "Hebrews", "James",
    "1 Peter", "2 Peter", "1 John", "2 John", "3 John",
    "Jude", "Revelation"
]
    escaped_book_names = [re.escape(book) for book in book_names]
    book_names_pattern = "|".join(escaped_book_names)

    pattern = rf"""
        (?:^|\b)                    # Start of string or word boundary
        ({book_names_pattern})      # Match any book name from the list
        \s+                         # Space between book name and numbers
        (\d+)                       # Chapter number (capture group 2)
        :                           # Colon separator
        (\d+)                       # Verse number (capture group 3)
        (?![\d:])                   # Negative lookahead to prevent partial matches
    """
    
    matches = re.finditer(pattern, text, flags=re.X | re.IGNORECASE)
    
    references = []
    seen = set()
    
    for match in matches:
        
        book_name = match.group(1)
        chapter = match.group(2)
        verse = match.group(3)
        
        full_ref = f"{book_name} {chapter}:{verse}"
        full_ref = re.sub(r'\s+', ' ', full_ref).strip()  
        
        if full_ref not in seen:
            seen.add(full_ref)
            references.append(full_ref)
    
    return references


def load_bible_data(file_path):
    df = pd.read_csv(file_path)
    documents = [
        Document(
            page_content=row["text"],
            metadata={
                "reference": row["reference"],
                "book": row["book"],
                "chapter": row["chapter"],
                "verse": row["verse"]
            }
        )
        for _, row in df.iterrows()
    ]
    return documents


def save_faiss_index(vector_store, save_path):
    vector_store.save_local(save_path)


def load_faiss_index(load_path):
    if os.path.exists(load_path):
        return FAISS.load_local(load_path, EMBEDDING_MODEL, allow_dangerous_deserialization=True)
    return None


def create_and_save_vector_store(documents, save_path):
    vector_store = FAISS.from_documents(documents, EMBEDDING_MODEL)
    save_faiss_index(vector_store, save_path)
    return vector_store

def initialize_system():
    vector_store = load_faiss_index(FAISS_INDEX_PATH)
    if vector_store is None:
        documents = load_bible_data(BIBLE_CSV_PATH)
        vector_store = create_and_save_vector_store(documents, FAISS_INDEX_PATH)
    return vector_store

def find_related_documents(query, vector_store):
    return vector_store.similarity_search(query, k=3)

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([f"{doc.metadata['reference']}: {doc.page_content}" for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    answer = response_chain.invoke({"user_query": user_query, "document_context": context_text})
    
    return answer, context_text 

st.set_page_config(
    page_title="Bible Scholar AI",
    page_icon="📖",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stChatInput input {
        border-radius: 20px !important;
        padding: 12px !important;
    }
    .stChatMessage {
        border-radius: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .verse-card {
        padding: 15px;
        background: white;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .sidebar .sidebar-content {
        background-color: #fff;
    }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.title("About Bible Scholar AI")
    st.markdown("""
    **📖 Bible Scholar AI**  
    A state-of-the-art Retrieval Augmented Generation system powered by Meta Llama 3 Instruct, designed to deliver precise, context-aware insights from the Bible.

    **How it Works**:  
    - Utilizes **Multi-QA-MPNET-Base-COS-V1** for embedding and FAISS for efficient vector database search to find relevant Bible verses based on your query.  
    - Generates intelligent, accurate answers backed by **Meta Llama 3 Instruct**.  
    - Provides scriptural references for further verification and in-depth study.
    """)


col1, col2 = st.columns([3, 1])
with col1:
    st.title("Bible Scholar AI 📖")
    st.markdown("Ask questions about the Bible and get accurate answers.")
    st.markdown("---")

vector_store = initialize_system()

if "messages" not in st.session_state:
    st.session_state.messages = []
    

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="📖" if message["role"] == "assistant" else None):
        st.markdown(message["content"])

user_input = st.chat_input("Ask your biblical question...")

if user_input:
   
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
        
    
    with st.spinner("🔍 Searching Scriptures..."):
        relevant_docs = find_related_documents(user_input, vector_store)
        ai_response, context_text = generate_answer(user_input, relevant_docs)
        references = find_bible_references(ai_response)
        processed_references = []
        for ref in references:
        
            parts = ref.split()
            
            book_name = "".join(parts[:-1]) 
            chapter_verse = parts[-1] 
            
            processed_ref = f"{book_name} {chapter_verse}".lower()
            processed_references.append(processed_ref)
        

    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    
   
    with st.chat_message("assistant", avatar="📖"):
        st.markdown(ai_response)
        
       
        with st.expander("📚 References"):
            df=pd.read_csv('bible_verses.csv')
            verses=[]
            for ref,pref in zip(references,processed_references):
                
                match = df[df['reference'] == pref]
                
                if not match.empty:
                    verse_text = match['text'].values[0]  
                    verses.append(f"**{ref}**: {verse_text}") 
                else:
                    verses.append(f"**{ref}**: Verse not found")
            st.markdown("\n\n".join(verses))
            

st.markdown(
    """
    <div style="position: fixed; bottom: 10px; right: 10px; opacity: 0.3;">
        <h4>Bible Scholar AI</h4>
    </div>
    """,
    unsafe_allow_html=True
)