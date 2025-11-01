import os
import streamlit as st
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile
import warnings
import time # Import time to simulate loading (for better user experience)

# Suppress minor warnings for a cleaner Streamlit app
warnings.filterwarnings("ignore")

# --- RAG Libraries ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama 

# --- SETUP: Load Environment Variables ---
load_dotenv()

# --- 1. RAG PIPELINE FUNCTION (Cached Backend) ---
@st.cache_resource(show_spinner="1. Processing PDF and Creating Vector Index...")
def process_pdf_and_create_vectorstore(uploaded_file):
    # This code successfully loads, splits, and embeds your PDF
    
    # Simulate loading bar for 1 second (optional, for better UX)
    time.sleep(1) 
    
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
        
    loader = PyPDFLoader(tmp_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(data)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    os.remove(tmp_path) # Clean up temp file
    return vectorstore

# Initialize chat history and QA chain in session state
def initialize_session_state(llm, vectorstore):
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": "Welcome! Please upload your PDF on the sidebar to begin chatting."}]
    
    if "qa_chain" not in st.session_state:
        # 1. Custom Prompt (Ensuring the AI stays on-topic and doesn't guess)
        custom_prompt_template = """You are a helpful and accurate RAG system. Answer the user's question based ONLY on the following context.
        If the answer is not present in the context, state clearly, "I can only answer based on the provided document and the answer is not present in the text."
        Provide a concise and factual answer.
        {context}
        Question: {question}
        """
        CUSTOM_PROMPT = PromptTemplate(
            template=custom_prompt_template,
            input_variables=["context", "question"]
        )

        # 2. Create the Final RAG Chain (Using the speedy k=5 fix)
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}), # <-- SPEED FIX: k=5
            return_source_documents=True,
            chain_type_kwargs={"prompt": CUSTOM_PROMPT}
        )

# Function to handle the RAG query and update history
def handle_query(query):
    # Store the user's message immediately
    st.session_state.messages.append({"role": "user", "content": query})
    
    qa_chain = st.session_state.qa_chain
    
    # Invoke the QA Chain
    result = qa_chain.invoke({"query": query})

    # Prepare the source string for verification feature
    sources = set()
    for doc in result['source_documents']:
        page = doc.metadata.get('page', 0)
        sources.add(f"Page {page + 1}")
    
    source_text = f"**Sources:** {', '.join(sorted(list(sources)))}"

    # Store the AI's response and sources
    st.session_state.messages.append({"role": "assistant", "content": result['result']})
    st.session_state.messages.append({"role": "source", "content": source_text})


def main():
    st.set_page_config(page_title="PDF Chatbot", layout="centered")
    st.markdown("<h1 style='text-align: center; color: #1e90ff;'>📚 Resume-Worthy RAG Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #aaa;'>Built with LangChain + Ollama (100% Stable & Local)</p>", unsafe_allow_html=True)
    st.markdown("---")

    # File Uploader
    uploaded_file = st.sidebar.file_uploader("1. Upload a PDF document:", type="pdf")

    if uploaded_file is not None:
        vectorstore = process_pdf_and_create_vectorstore(uploaded_file)
        
        initialize_session_state(Ollama(model="llama2:7b", temperature=0.1), vectorstore)
        st.sidebar.success("Index ready! Start chatting.")


        for message in st.session_state.messages:
            if message["role"] == "source":
                st.info(message["content"]) # Display source information
            elif message["role"] == "user":
                with st.chat_message("user", avatar="🙋‍♂️"):
                    st.markdown(message["content"])
            elif message["role"] == "assistant":
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(message["content"])
        
        # User Input (Allows continuous conversation)
        if query := st.chat_input("Ask a question about the document..."):
            handle_query(query)
            st.rerun() # Rerun to display the new messages immediately


if __name__ == "__main__":
    main()