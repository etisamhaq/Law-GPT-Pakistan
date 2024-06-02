import streamlit as st
import cassio
from PyPDF2 import PdfReader
from langchain_community.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

# Initialize the connection to your database
ASTRA_DB_APPLICATION_TOKEN = "YOUR ASTRA_DB_APPLICATION_TOKEN"
ASTRA_DB_ID = "YOUR ASTRA_DB_ID"
OPENAI_API_KEY = "YOUR OPENAI_API_KEY"

cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# Create the LangChain embedding and LLM objects
llm = OpenAI(openai_api_key=OPENAI_API_KEY)
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Create the vector store backed by Astra DB
astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="qa_mini_demo",
    session=None,
    keyspace=None,
)

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ''
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text

# Function to split text
def split_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_text(text)

# Streamlit app
def main():
    st.title("LawGPT - The Ultimate Legal AI on Pakistani Laws and Constitution")

    # Path to the PDF file
    pdf_path = "Pakistan.pdf"

    # Extract text from the PDF
    raw_text = extract_text_from_pdf(pdf_path)

    # Split the text into chunks
    texts = split_text(raw_text)

    # Load the dataset into the vector store
    astra_vector_store.add_texts(texts)
    astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

    # st.subheader("Extracted Text from PDF")
    # st.text_area("PDF Text", raw_text, height=300)

    # QA section
    st.subheader("Ask a Question")
    query_text = st.text_input("Enter your question:")

    if query_text:
        answer = astra_vector_index.query(query_text, llm=llm).strip()
        st.write(f"**ANSWER:** {answer}")

        st.write("**FIRST DOCUMENTS BY RELEVANCE:**")
        for doc, score in astra_vector_store.similarity_search_with_score(query_text, k=1):
            st.write(f"[{score:.4f}] {doc.page_content[:84]}...")

if __name__ == "__main__":
    main()

