from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import os
from rag_logic import RagEngine
from memory import RedisMemory
from langchain_core.messages import HumanMessage, AIMessage

# Initialize Web App
app = FastAPI(title="DevOps RAG Assistant")

# Initialize Engine and Memory once at startup
engine = RagEngine()
memory = RedisMemory()

class ChatRequest(BaseModel):
    user_id: str
    message: str

# The Website UI (HTML/CSS/JS)
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DevOps AI Assistant</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #1e1e1e; color: #d4d4d4; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
        .chat-container { width: 100%; max-width: 800px; background-color: #252526; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); display: flex; flex-direction: column; height: 85vh; overflow: hidden; }
        .header { background-color: #333333; padding: 15px 20px; font-size: 1.2rem; font-weight: bold; border-bottom: 1px solid #444; }
        .chat-box { flex: 1; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 15px; }
        .msg { max-width: 80%; padding: 12px 16px; border-radius: 8px; line-height: 1.5; word-wrap: break-word; white-space: pre-wrap; }
        .user-msg { align-self: flex-end; background-color: #0e639c; color: white; }
        .bot-msg { align-self: flex-start; background-color: #3c3c3c; color: #d4d4d4; }
        .input-area { padding: 15px; background-color: #333333; display: flex; gap: 10px; border-top: 1px solid #444; }
        input[type="text"] { flex: 1; padding: 12px; background-color: #3c3c3c; border: 1px solid #555; border-radius: 6px; color: white; font-size: 1rem; outline: none; }
        input[type="text"]:focus { border-color: #0e639c; }
        button { padding: 12px 20px; background-color: #0e639c; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 1rem; font-weight: bold; transition: 0.2s; }
        button:hover { background-color: #1177bb; }
        .loading { font-style: italic; color: #888; align-self: flex-start; margin-left: 10px; }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="header">🚀 DevOps RAG Assistant</div>
        <div class="chat-box" id="chatBox">
            <div class="msg bot-msg">Hello! I am Nano, your internal DevOps assistant. How can I help you today?</div>
        </div>
        <div class="input-area">
            <input type="text" id="userInput" placeholder="Ask about incidents, servers, credentials..." onkeypress="handleEnter(event)">
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        // Generate a random session ID for this browser tab
        const userId = "web_user_" + Math.random().toString(36).substring(7);
        const chatBox = document.getElementById('chatBox');
        const userInput = document.getElementById('userInput');

        function appendMessage(text, isUser) {
            const div = document.createElement('div');
            div.className = `msg ${isUser ? 'user-msg' : 'bot-msg'}`;
            div.innerText = text;
            chatBox.appendChild(div);
            chatBox.scrollTop = chatBox.scrollHeight;
            return div;
        }

        async function sendMessage() {
            const text = userInput.value.trim();
            if (!text) return;

            appendMessage(text, true);
            userInput.value = '';

            const loadingIndicator = document.createElement('div');
            loadingIndicator.className = 'loading';
            loadingIndicator.innerText = 'Thinking...';
            chatBox.appendChild(loadingIndicator);
            chatBox.scrollTop = chatBox.scrollHeight;

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ user_id: userId, message: text })
                });
                
                const data = await response.json();
                chatBox.removeChild(loadingIndicator);
                appendMessage(data.reply, false);
            } catch (error) {
                chatBox.removeChild(loadingIndicator);
                appendMessage('Error: Unable to connect to the server.', false);
            }
        }

        function handleEnter(event) {
            if (event.key === 'Enter') sendMessage();
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def serve_website():
    """Serves the front-end HTML interface."""
    return html_content

@app.post("/chat")
async def handle_chat(request: ChatRequest):
    """API Endpoint to process messages through the RAG Engine."""
    try:
        # Fetch History
        history = memory.get_messages(request.user_id)
        
        # Get Response
        response_text = engine.get_rag_response(request.message, history)
        
        # Save History
        memory.add_message(request.user_id, HumanMessage(content=request.message))
        memory.add_message(request.user_id, AIMessage(content=response_text))
        
        return {"reply": response_text}
    except Exception as e:
        return {"reply": f"System Error: {str(e)}"}

if __name__ == "__main__":
    # Runs the web server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
