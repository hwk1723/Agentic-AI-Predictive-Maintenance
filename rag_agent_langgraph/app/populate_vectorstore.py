import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
persist_directory = r"/Users/howaikit/Documents/GitHub/Agentic-AI-Predictive-Maintenance/rag_agent_langgraph/app/vectorstore"
collection_name = "maintenance_and_sop_collection"
pdf_path = "/Users/howaikit/Documents/GitHub/Agentic-AI-Predictive-Maintenance/rag_agent_langgraph/app/Document.pdf"

if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)

try:
    pages = pdf_loader.load()
    print(f"Loaded {len(pages)} pages from the PDF file.")
except Exception as e:
    print(f"Error loading PDF file: {e}")
    raise

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200)

pages_split = text_splitter.split_documents(pages)


vector_store = Chroma.from_documents(
    documents=pages_split,
    embedding=embeddings,
    persist_directory=persist_directory,
    collection_name=collection_name
)

# vector_store.persist()