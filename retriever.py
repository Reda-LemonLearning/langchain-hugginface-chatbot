import os
from dotenv import load_dotenv
from data import Chroma,HuggingFaceEmbeddings

load_dotenv()
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_ID")
DATA_CHROMA_PATH = os.getenv("DATA_CHROMA_PATH")

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    multi_process=True,
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
)
documents_vector_db = Chroma(
    persist_directory=DATA_CHROMA_PATH,
    embedding_function=embedding_model,
)
documents_retriever  = documents_vector_db.as_retriever(k=10)