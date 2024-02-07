from getpass import getpass
import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

OPENAI_API_KEY = getpass("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")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# If you need to specify your organization ID
# OPENAI_ORGANIZATION = getpass("Enter your OpenAI Organization ID: ")
# os.environ["OPENAI_ORGANIZATION"] = OPENAI_ORGANIZATION
template = """
Question: {question}

Answer: This is insights from data.
"""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm = OpenAI(openai_api_key=OPENAI_API_KEY)

file_path = "data/Aging report.csv"
loader = CSVLoader(file_path=file_path, encoding="utf-8", csv_args={'delimiter': ','})
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
text_chunks = text_splitter.split_documents(data)

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
docsearch = FAISS.from_documents(text_chunks, embeddings)

llm_chain = LLMChain(prompt=prompt, llm=llm, retriever=docsearch.as_retriever())

question = "give very important insights"
llm_chain.run(question)
