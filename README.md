# PDF Chat 🤖

A Streamlit-based application that allows users to have interactive conversations with their PDF documents using GPT and RAG (Retrieval Augmented Generation) technique.

## 🌟 Features

- Upload multiple PDF documents
- Interactive chat interface
- RAG (Retrieval Augmented Generation) for accurate responses
- Document chunking for better context handling
- Conversation memory to maintain context
- Clean and intuitive user interface

## 🛠️ Technologies Used

- [Streamlit](https://streamlit.io/) - Web application framework
- [LangChain](https://python.langchain.com/) - LLM application framework
- [OpenAI GPT](https://openai.com/) - Large Language Model
- [FAISS](https://github.com/facebookresearch/faiss) - Vector store for embeddings
- [PyPDF](https://pypdf2.readthedocs.io/) - PDF processing

## 📋 Prerequisites

- Python 3.7+
- OpenAI API key

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/rajeshai/Streamlit-RAG-chatwith-PDF.git
cd Streamlit-RAG-chatwith-PDF
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create a `requirements.txt` file with the following dependencies:
```
streamlit
langchain
openai
pypdf
tiktoken
langchain-community
faiss-cpu
```

## 💻 Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (typically `http://localhost:8501`)

3. Enter your OpenAI API key in the sidebar

4. Upload one or more PDF files

5. Click "Process PDFs" to initialize the chat system

6. Start asking questions about your documents!

## 🔧 How It Works

1. **Document Processing**:
   - PDFs are loaded and split into smaller chunks
   - Text chunks are converted into embeddings using OpenAI's embedding model
   - Embeddings are stored in a FAISS vector store for efficient retrieval

2. **Query Processing**:
   - User questions are processed to find relevant document chunks
   - Retrieved context is combined with the question
   - GPT generates accurate responses based on the provided context

3. **Conversation Management**:
   - Maintains chat history for context
   - Handles multiple PDF documents simultaneously
   - Provides a clean chat interface for interaction

## ⚠️ Important Notes

- Requires an OpenAI API key (paid service)
- Large PDF files may take longer to process
- Processing time depends on the number and size of uploaded PDFs

## 🤝 Contributing

Feel free to open issues or submit pull requests if you have suggestions for improvements!


## 🙏 Acknowledgments

- Built using OpenAI's GPT model
- Inspired by the LangChain framework
- Made possible by the Streamlit community
