import os
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader

def create_sidebar():
    with st.sidebar:
        st.title("PDF Chat")
        st.markdown("### Quick Demo of RAG")
        
        api_key = st.text_input("OpenAI API Key:", type="password")
        
        st.markdown("""
        ### Tools Used
        • OpenAI
        • LangChain
        • ChromaDB
        
        ### Steps
        1. Add API key
        2. Upload PDF
        3. Chat!
        """)
        
        return api_key

def save_uploaded_file(uploaded_file, path='./uploads/'):
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

@st.cache_data
def load_texts_from_papers(papers):
    all_texts = []
    for paper in papers:
        file_path = save_uploaded_file(paper)
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        all_texts.extend(texts)
        os.remove(file_path)
    return all_texts

@st.cache_resource
def initialize_vectorstore():
    embedding = OpenAIEmbeddings(openai_api_key=st.session_state.api_key)
    vectorstore = Chroma(embedding_function=embedding, persist_directory="db")
    return vectorstore

def main():
    st.set_page_config(page_title="PDF Chat", layout="wide")

    # Get API key from sidebar
    api_key = create_sidebar()
    
    if api_key:
        st.session_state.api_key = api_key

    st.title("Chat with PDF")
    papers = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not api_key:
        st.warning("Please enter your OpenAI API key")
        return

    try:
        vectorstore = initialize_vectorstore()
        texts = load_texts_from_papers(papers) if papers else []
        
        if texts:
            vectorstore.add_documents(texts)
            qa_chain = ConversationalRetrievalChain.from_llm(
                ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
                vectorstore.as_retriever(),
                memory=ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True
                )
            )
        else:
            memory = ConversationBufferMemory(memory_key="chat_history")
            qa_chain = ConversationChain(
                llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
                memory=memory
            )
        
        # Chat interface
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("Ask about your PDFs"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                try:
                    if texts:
                        result = qa_chain({"question": prompt})
                        response = result["answer"]
                    else:
                        result = qa_chain.predict(input=prompt)
                        response = result
                    
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.markdown(response)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                
    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
