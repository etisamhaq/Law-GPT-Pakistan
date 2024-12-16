import os
import PyPDF2
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# Set API key from Streamlit secrets
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def create_qa_system():
    # Use Pakistan.pdf directly
    pdf_path = "Pakistan.pdf"
    pdf_text = extract_text_from_pdf(pdf_path)

    # Split text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(pdf_text)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings()

    # Create vector store
    db = Chroma.from_texts(texts, embeddings)

    # Initialize Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
    )

    # Create a retrieval chain
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever(search_kwargs={"k": 1}))

    return qa

def main():
    st.title("Law-GPT Chatbot")
    st.write("Ask questions about Pakistan's Constitution and Legal System")

    # Initialize session state for QA system if not already done
    if 'qa_system' not in st.session_state:
        with st.spinner('Initializing the chatbot...'):
            st.session_state.qa_system = create_qa_system()
        st.success('Chatbot is ready!')

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask your question about Pakistan's legal system"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner('Thinking...'):
                response = st.session_state.qa_system.run(prompt)
            st.write(response)
            
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()