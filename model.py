import streamlit as st
import os
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain

def main():
    st.set_page_config(page_title="Data Insighter")
    st.title("Data Insighter")
    
    DB_FAISS_PATH = "vectorstore/db_faiss"
    TEMP_DIR = "temp"

    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    uploaded_file = "data/dataset.csv"
    print("Uploaded File:", uploaded_file)

    
    if uploaded_file is not None:
        st.write(f"Using CSV file: {uploaded_file}")
        file_path = uploaded_file
        print("File Path:", file_path)

        st.write("Processing CSV file...")
        loader = CSVLoader(file_path=file_path, encoding="utf-8", csv_args={'delimiter': ','})
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        text_chunks = text_splitter.split_documents(data)

        st.write(f"Total text chunks: {len(text_chunks)}")

        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        docsearch = FAISS.from_documents(text_chunks, embeddings)
        docsearch.save_local(DB_FAISS_PATH)

        llm = CTransformers(model="models/llama-2-7b-chat.ggmlv3.q4_0.bin",
                            model_type="llama",
                            max_new_tokens=512,
                            temperature=0.1)

        qa = ConversationalRetrievalChain.from_llm(llm, retriever=docsearch.as_retriever())

        st.write("Enter your query:")
        query = st.text_input("Input Prompt:")
        if query:
            with st.spinner("Processing your question..."):
                chat_history = []
                result = qa.invoke({"question": query, "chat_history": chat_history})
                st.write("Response:", result['answer'])

if __name__ == "__main__":
    main()
