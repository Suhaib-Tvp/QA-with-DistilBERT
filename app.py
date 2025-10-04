import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from PyPDF2 import PdfReader
import os

# Load API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="PDF QA with LangChain", layout="wide")
st.title("📄 PDF Question Answering App")

# Sidebar upload
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    # Read PDF
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_text(text)
    st.success(f"PDF loaded and split into {len(chunks)} chunks!")

    # Create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    st.success("Embeddings created and stored in FAISS vector store!")

    # User question
    query = st.text_input("Ask a question about your PDF:")
    if query:
        docs = vectorstore.similarity_search(query, k=5)
        llm = OpenAI(openai_api_key=OPENAI_API_KEY)
        qa_chain = load_qa_chain(llm, chain_type="stuff")
        answer = qa_chain.run(input_documents=docs, question=query)
        st.markdown(f"**Answer:** {answer}")
