import os
from dotenv import load_dotenv
from config import SYSTEM_PROMPT1, SYSTEM_PROMPT2
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_qdrant import QdrantVectorStore
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from qdrant_client import QdrantClient
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
import memory

load_dotenv()

# chatHistory = [
#     HumanMessage(content="Hello How are you ChatBot?"),
#     AIMessage(content="I am fine! How are you?"),
#     HumanMessage(content="I am great, thaks for asking."),
#     AIMessage(content="That is amazing. How may I assist you today?"),
#     HumanMessage(content="I have a question about INCIDENT #9902."),
#     AIMessage(content="What would you like to know?"),
#     HumanMessage(content="sdadjo"),
#     AIMessage(content="I couldn't understand you.")
# ]

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


class RagEngine:
    def __init__(self):
        
        
        self.groqKey = os.getenv("Ai_Api_Key")
        self.llm = ChatGroq(
            api_key=self.groqKey,
            model="llama-3.1-8b-instant",
            temperature=0.5
        )
        
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
            ("system",SYSTEM_PROMPT1),
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
        

    def outputFormaterandChecker(self,paragraph):
        paragraph = paragraph.split("\n")
        # print("\n--- DEBUG: ITEMS FROM LLM ---")
        # print(f"The paragraph is: {paragraph}")
        # print("----------------------------------\n")
        if paragraph[0] != "NO_RAG" and paragraph[0] != "":
            return paragraph
        return None

    def retrieverFormaterandChecker(self,items):
        if items == None:
            # print("\n--- DEBUG: ITEMS SENT TO DATABASE ---")
            # print("No Context In Database, answer from General Knowledge.")
            # print("----------------------------------\n")
            return "No Context In Database, answer from General Knowledge."
        
        # print("\n--- DEBUG: ITEMS SENT TO DATABASE ---")
        # print(items)
        # print("----------------------------------\n")
        doc = self.retriever.map().invoke(items)
        payloads = "\n".join([x.page_content for y in doc for x in y])
        # print("\n--- DEBUG: CONTEXT SENT TO LLM ---")
        # print(payloads)
        # print("----------------------------------\n")
        return payloads    


# def main():
#     try:
#         engine = RagEngine()
#     except Exception as e:
#         print(f"{Colors.FAIL}[CRITICAL] Failed to initialize Engine: {e}{Colors.ENDC}")
#         sys.exit(1)

#     # Local session history
#     chat_history = []

#     print(f"{Colors.BOLD}--- LOGIC-ROUTER RAG CLI v1.0 ---{Colors.ENDC}")
#     print("Type 'exit' or 'quit' to terminate session.\n")

#     while True:
#         try:
#             user_input = input(f"{Colors.GREEN}You:{Colors.ENDC} ").strip()
            
#             if not user_input:
#                 continue
                
#             if user_input.lower() in ["exit", "quit"]:
#                 print(f"\n{Colors.BLUE}[INFO]{Colors.ENDC} Shutting down session...")
#                 break

#             # Generate Response
#             print(f"{Colors.WARNING}Thinking...{Colors.ENDC}", end="\r")
#             response = engine.get_rag_response(user_input, chat_history)
            
#             # Clear "Thinking..." line
#             print(" " * 20, end="\r")
            
#             # Print Bot Response
#             print(f"{Colors.BLUE}Bot:{Colors.ENDC} {response}\n")

#             # Update History
#             chat_history.append(HumanMessage(content=user_input))
#             chat_history.append(AIMessage(content=response))
            
#             # Keep history manageable (last 10 turns)
#             if len(chat_history) > 20:
#                 chat_history = chat_history[-20:]

#         except KeyboardInterrupt:
#             print(f"\n{Colors.BLUE}[INFO]{Colors.ENDC} Session interrupted.")
#             break
#         except Exception as e:
#             print(f"\n{Colors.FAIL}[ERROR]{Colors.ENDC} {e}")

# if __name__ == "__main__":
#     main()
 

