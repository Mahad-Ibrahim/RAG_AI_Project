import os
import redis
import json
from langchain_core.messages import HumanMessage, AIMessage

class RedisMemory:
    def __init__(self):
        # Securely grab the Redis hostname from the Docker environment variables
        redis_host = os.getenv("REDIS_HOST", "redis-cache")
        
        # Initialize the connection directly to the Docker container
        self.client = redis.Redis(host=redis_host, port=6379, db=0, decode_responses=True)

    def get_messages(self, user_id):
        try:
            # lrange retrieves the chat history list for the specific user
            raw_messages = self.client.lrange(user_id, 0, -1)
            messages = []
            for msg in raw_messages:
                data = json.loads(msg)
                if data["type"] == "human":
                    messages.append(HumanMessage(content=data["content"]))
                else:
                    messages.append(AIMessage(content=data["content"]))
            return messages
        except Exception:
            # If Redis is temporarily unavailable, return an empty history instead of crashing
            return []

    def add_message(self, user_id, message):
        try:
            data = {
                "type": "human" if isinstance(message, HumanMessage) else "ai", 
                "content": message.content
            }
            # rpush adds the new message to the end of the user's list
            self.client.rpush(user_id, json.dumps(data))
            
            # Keep only the last 10 messages so the AI context window doesn't overflow
            self.client.ltrim(user_id, -10, -1)
        except Exception as e:
            print(f"Redis Error: {e}")
