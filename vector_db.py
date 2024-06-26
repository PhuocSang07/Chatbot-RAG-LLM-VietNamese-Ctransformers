from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ai21 import AI21SemanticTextSplitter
from dotenv import load_dotenv
import re
import os

load_dotenv()


pdf_data_path = './documents'
vector_db_path = './db'
model_name = 'bkai-foundation-models/vietnamese-bi-encoder'
AI21_TOKEN = os.getenv('AI21_TOKEN')
os.environ["AI21_API_KEY"] = AI21_TOKEN


def clean_text(text):
    text = re.sub(r'[^\w\s,.-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace(" \n", "\n").replace("\n ", "\n").replace("\n", "\n\n")
    
    return text

def create_db_from_files():
    loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls = PyPDFLoader)
    documents = loader.load()

    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    text_splitter = AI21SemanticTextSplitter(chunk_size=1024, chunk_overlap=128)

    chunks = text_splitter.split_documents(documents)

    for chunk in chunks:
        chunk.page_content = clean_text(chunk.page_content)

    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
)

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(vector_db_path)
    return db

create_db_from_files()