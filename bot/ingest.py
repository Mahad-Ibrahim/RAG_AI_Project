import os
import glob

# Only import basic system modules at the top
def ingest_data():
    # Detect CI mode
    is_ci = os.getenv("CI_FAST_MODE") == "true"

    # HEAVY IMPORTS: Only loaded if we actually need them
    from langchain_core.documents import Document
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import FastEmbedEmbeddings
    from langchain_core.embeddings import FakeEmbeddings
    from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
    from qdrant_client import QdrantClient, models

    COLLECTION_NAME = "knowledge_base"

    # Setup Embeddings
    if is_ci:
        embeddings = FakeEmbeddings(size=384)
    else:
        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    
    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant-db:6333")
    client = QdrantClient(url=qdrant_url)
    
    # Setup Collection
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

    if is_ci:
        print("Injecting dummy test data...")
        test_chunk = [Document(page_content="INCIDENT #8821: The resolution is fixed and the new key is sk_live_998877_updated.", metadata={})]
        vector_store.add_documents(test_chunk)
    else:
        files = glob.glob("Data/**/*.txt", recursive=True)
        split_text = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50)
        for file_path in files:
            try:
                chunks = split_text.split_documents(TextLoader(file_path).load())
                vector_store.add_documents(chunks)
            except Exception as e:
                print(f"Skipping {file_path}: {e}")

    print("Success.")

if __name__ == "__main__":
    ingest_data()
