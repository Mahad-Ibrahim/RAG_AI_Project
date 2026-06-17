from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient, models
import os

COLLECTION_NAME = "knowledge_base"

def ingest_data():
    documents = []
    
    txt_loader = DirectoryLoader("Data", glob="**/*.txt", loader_cls=TextLoader)
    documents.extend(txt_loader.load())

    if not documents:
        print("Error: No Documents found in current Directory.")
        return
    
    split_text = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50, separators=["\n\n","\n"])

    chunks = split_text.split_documents(documents)

    embeddings = FastEmbedEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )
    
    sparse_embeddings = FastEmbedSparse(
        model_name="Qdrant/bm25"
    )
    
    print(f"Sparse Embeddings is {type(sparse_embeddings)}")
    
    try:
        
        client = QdrantClient(url="http://localhost:6333")
        
        if client.collection_exists(COLLECTION_NAME):
            client.delete_collection(COLLECTION_NAME)
        
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=384, # BGE-Small dimension
                distance=models.Distance.COSINE
            ),
            # This enables the "Keyword Search" capability
            sparse_vectors_config={
                "langchain-sparse": models.SparseVectorParams()
            }
        )
        

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
            # Explicitly pass sparse embedding here
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID
        )
        
        vector_store.add_documents(chunks)

        print("Success. Data was inserted successfully.")
    
    
    except Exception as e:
        print(f"Error during ingestion: {e}")
        return



if __name__ == "__main__":
    ingest_data()
