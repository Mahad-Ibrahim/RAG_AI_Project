import os
import glob
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.embeddings import FakeEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient, models

COLLECTION_NAME = "knowledge_base"

def ingest_data():
    # Detect if we are in CI to save memory
    is_ci = os.getenv("CI") == "true"
    
    if is_ci:
        print("CI detected: Using FakeEmbeddings to save memory.")
        embeddings = FakeEmbeddings(size=384)
    else:
        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    
    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant-db:6333")
    client = QdrantClient(url=qdrant_url)
    
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
        sparse_vectors_config={"langchain-sparse": models.SparseVectorParams()}
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID
    )

    files = glob.glob("Data/**/*.txt", recursive=True)
    if not files:
        print("Error: No Documents found.")
        return

    split_text = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50, separators=["\n\n","\n"])

    for file_path in files:
        try:
            print(f"Ingesting: {file_path}")
            loader = TextLoader(file_path)
            docs = loader.load()
            chunks = split_text.split_documents(docs)
            vector_store.add_documents(chunks)
        except Exception as e:
            print(f"Skipping {file_path} due to error: {e}")

    print("Success. Data was inserted successfully.")

if __name__ == "__main__":
    ingest_data()
