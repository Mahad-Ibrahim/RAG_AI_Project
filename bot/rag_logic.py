import os
from dotenv import load_dotenv
from config import SYSTEM_PROMPT1, SYSTEM_PROMPT2
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
from langchain_core.messages import HumanMessage, AIMessage
from qdrant_client import QdrantClient
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_qdrant import FastEmbedSparse, RetrievalMode
import memory

load_dotenv()

class RagEngine:
    def __init__(self):
        # Fallback to standard GROQ_API_KEY if Ai_Api_Key is not found
        self.groqKey = os.getenv("Ai_Api_Key") or os.getenv("GROQ_API_KEY")
        
        if not self.groqKey:
            raise ValueError("No API Key found. Please set Ai_Api_Key or GROQ_API_KEY.")

        self.llm = ChatGroq(
            api_key=self.groqKey,
            model="llama-3.1-8b-instant",
            temperature=0.5
        )
    
        is_ci = os.getenv("CI_FAST_MODE") == "true"
        
        if is_ci:
            from langchain_core.embeddings import FakeEmbeddings
            self.embeddings = FakeEmbeddings(size=384)
        else:
            from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
            self.embeddings = FastEmbedEmbeddings(
                model_name="BAAI/bge-small-en-v1.5"
            )
        
        self.sparse_embeddings = FastEmbedSparse(
            model_name="Qdrant/bm25"
        )
        
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrantClient = QdrantClient(url=qdrant_url)
        
        vectorStorage = QdrantVectorStore(
            client=self.qdrantClient,
            collection_name="knowledge_base",
            embedding=self.embeddings,
            sparse_embedding=self.sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID
        )
        
        self.retriever = vectorStorage.as_retriever(
            search_type="similarity",
            search_kwargs={"k":3}
        )
        
        self.outputFaC = RunnableLambda(self.outputFormaterandChecker)
        self.retrieverFaC = RunnableLambda(self.retrieverFormaterandChecker)

        self.redisMemory = memory.RedisMemory()

    def get_rag_response(self, question, chat_history):
        overallQuery = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT1),
            ("placeholder", "{chatHistory}"),
            ("human", "{Question}") 
        ]) 
        
        chain = overallQuery | self.llm | StrOutputParser() | self.outputFaC | self.retrieverFaC

        finalPromptPayloads = chain.invoke({
            "chatHistory": chat_history,
            "Question": question
        })
        
        finalPrompt = SYSTEM_PROMPT2 + f"\n\n[DATABASE CONTEXT]:\n{finalPromptPayloads}"
        
        finalQuery = ChatPromptTemplate.from_messages([
            ("system", finalPrompt),
            ("placeholder", "{chatHistory}"),
            ("human", "{Question}") 
        ])
        
        finalChain = finalQuery | self.llm | StrOutputParser()
        
        output = finalChain.invoke({
            "chatHistory": chat_history,
            "Question": question
        })
        
        return output

    def outputFormaterandChecker(self, paragraph):
        paragraph = paragraph.split("\n")
        if paragraph[0] != "NO_RAG" and paragraph[0] != "":
            return paragraph
        return None

    def retrieverFormaterandChecker(self, items):
        if items == None:
            return "No Context In Database, answer from General Knowledge."
        
        doc = self.retriever.map().invoke(items)
        payloads = "\n".join([x.page_content for y in doc for x in y])
        return payloads
