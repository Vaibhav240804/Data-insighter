import streamlit as st
import os
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from lida import Manager, TextGenerationConfig , llm  
# load .env

def main():
    st.set_page_config(page_title="Data Insighter")
    st.title("Data Insighter")
    
    DB_FAISS_PATH = "vectorstore/db_faiss"
    TEMP_DIR = "temp"
    # set openAI_API key in environment variable
    # os.environ["OPENAI_API"] = os.getenv("OPENAI_API")
    # st.secrets["openai_api"] = os.getenv("OPENAI_API")
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    uploaded_file = "data/Aging report.csv"
    print("Uploaded File:", uploaded_file)


    def plots(text):
        # goals can also be based on a persona
        lida = Manager()
        textgen_config = TextGenerationConfig(
            max_tokens=50,
            temperature=0.7,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        summary = lida.summarize("dummy.csv", textgen_config=textgen_config)
        persona = "a desk officer at fertilizers company who aims to minimize age of stock(i.e. selling generated product quickely, and not keeping it for too long) wants your help to navigate through given data such that he can make better decisions for his aim."
        personal_goals = lida.goals(summary, n=5, persona=persona, textgen_config=textgen_config)

        i = 0
        library = "seaborn"
        goals = personal_goals
        textgen_config = TextGenerationConfig(n=2, temperature=0.2, use_cache=True)

        st.write("Automated visualizing the data...")
  
        while i < len(goals):
            charts = lida.visualize(summary=summary, goal=goals[i], textgen_config=textgen_config, library=library)  
            st.write(charts)
            i += 1

        st.write("Generating the insights from given input...")       
        user_query = f"plot graphs for {text}"
        textgen_config = TextGenerationConfig(n=2, temperature=0.2)
        charts = lida.visualize(summary=summary, goal=user_query, textgen_config=textgen_config)
        for chart in charts:
            st.write(chart)
    
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
            # plots(query)
            with st.spinner("Processing your question..."):
                chat_history = []
                result = qa.invoke({"question": query, "chat_history": chat_history})
                st.write("Response:", result['answer'])

if __name__ == "__main__":
    main()
