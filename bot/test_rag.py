import os
import pytest
from rag_logic import RagEngine

def test_rag_knowledge_retrieval():
    """
    Tests if the RAG system successfully retrieves internal knowledge 
    and returns the expected output from the local database.
    """
    # Ensure the Groq API key is available in the environment
    assert "Ai_Api_Key" in os.environ, "Ai_Api_Key environment variable is missing"
    
    # Initialize the RAG Engine
    engine = RagEngine()
    
    # Query specific to internal data (Incident #8821)
    question = "What was the resolution and the new key for INCIDENT #8821?"
    
    # Empty chat history for a fresh query
    chat_history = []
    
    # Fetch AI response
    response = engine.get_rag_response(question, chat_history)
    
    # Verify the correct internal data was retrieved and formulated by the LLM
    expected_secret = "sk_live_998877_updated"
    
    assert expected_secret in response, f"RAG Bot failed to retrieve the correct API key. Bot output was: {response}"
