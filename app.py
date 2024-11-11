import os
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None

def create_sidebar():
    with st.sidebar:
        st.title("🤖 PDF Chat")
        
        api_key = st.text_input("OpenAI API Key:", type="password", help="Get your API key from OpenAI website")
        
        st.markdown("""
        ### What is this?
        A simple app that lets you chat with your PDF files using GPT and RAG.
        
        ### How to use
        1. Paste your OpenAI API key
        2. Upload PDF file(s)
        3. Click 'Process PDFs'
        4. Start asking questions!
        
        ### Built using
        - LangChain
        - OpenAI
        - FAISS
        - Streamlit
        
        Made with ☕
        """)
        
        return api_key

def process_pdfs(papers, api_key):
    """Process PDFs and return whether processing was successful"""
    if not papers:
        return False
        
    with st.spinner("Processing PDFs..."):
        try:
            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            all_texts = []
            
            for paper in papers:
                file_path = os.path.join('./uploads', paper.name)
                os.makedirs('./uploads', exist_ok=True)
                with open(file_path, "wb") as f:
                    f.write(paper.getbuffer())
                
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                )
                texts = text_splitter.split_documents(documents)
                all_texts.extend(texts)
                os.remove(file_path)
            
            vectorstore = FAISS.from_documents(all_texts, embeddings)
            
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            
            st.session_state.chain = ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=api_key),
                retriever=vectorstore.as_retriever(),
                memory=memory,
                return_source_documents=False,
                chain_type="stuff"
            )
            
            st.success(f"Processed {len(papers)} PDF(s) successfully!")
            return True
            
        except Exception as e:
            st.error(f"Error processing PDFs: {str(e)}")
            return False

def main():
    st.set_page_config(page_title="PDF Chat")
    
    api_key = create_sidebar()

    st.title("💬 Chat with your PDFs")
    st.markdown("""
    ### 👋 Hey there!
    This is a simple demo showing how to chat with your PDF documents using GPT and RAG (Retrieval Augmented Generation).
    
    #### Try it out:
    - Upload one or more PDFs 
    - Ask questions about their content
    - The app will use RAG to find relevant info and answer your questions
    """)
    
    st.divider()
    
    # File uploader with custom styling
    st.markdown("### 📄 Upload your documents")
    papers = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
    
    if papers:
        st.markdown(f"*{len(papers)} files uploaded*")
        if st.button("Process PDFs"):
            process_pdfs(papers, api_key)
    
    st.divider()
    
    if not api_key:
        st.warning("👆 Please enter your OpenAI API key in the sidebar to start")
        return
    
    # Chat interface
    st.markdown("### 💭 Chat")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask about your PDFs..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            if st.session_state.chain is None:
                response = "Please upload and process a PDF first! 👆"
            else:
                with st.spinner("Thinking..."):
                    result = st.session_state.chain({"question": prompt})
                    response = result["answer"]
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
