from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from lida import Manager, TextGenerationConfig

app = FastAPI()

class Query(BaseModel):
    text: str

DB_FAISS_PATH = "vectorstore/db_faiss"
TEMP_DIR = "temp"

@app.post("/process_csv/")
async def process_csv(query: Query):
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    uploaded_file = "data/Aging report.csv"
    print("Uploaded File:", uploaded_file)

    file_path = uploaded_file
    print("File Path:", file_path)

    loader = CSVLoader(file_path=file_path, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(data)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    docsearch = FAISS.from_documents(text_chunks, embeddings)
    docsearch.save_local(DB_FAISS_PATH)

    llm = CTransformers(model="models/llama-2-7b-chat.ggmlv3.q4_0.bin",
                        model_type="llama",
                        max_new_tokens=512,
                        temperature=0.1)

    qa = ConversationalRetrievalChain.from_llm(llm, retriever=docsearch.as_retriever())

    chat_history = []  # Initialize conversation history
    result = qa.invoke({"question": query.text, "chat_history": chat_history[-3:0]})

    return {"response": result['answer']}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
