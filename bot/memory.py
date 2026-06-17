import json
import redis
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

class RedisMemory:
    def __init__(self):
        self.ttl = 300
        self.port = 6379
        self.host = 'localhost'
        self.db = 0
        
        try:
            self.client = redis.Redis(
                db=self.db,
                decode_responses=True
            )
            
            self.client.ping()
            
        except Exception as e:
            print(f"Error connecting to Redis: {e}")
            self.client = None    
        
        
        
    def add_message(self, key, content: HumanMessage | AIMessage):
        
        msg = json.dumps({
            "type": "human" if isinstance(content, HumanMessage) else "ai",
            "content": content.content
        })
        
        self.client.rpush(key, msg)
        self.client.expire(key, self.ttl)
        
        if self.client.llen(key) > 20:
            self.client.lpop(key)
    
    def get_messages(self, key : json):
        message_json = self.client.lrange(key,0, -1)
        chat_history = []
        if not message_json:
            return []
        if message_json:
            for val in message_json:
                msg= json.loads(val)
                if msg["type"] == "human":
                    chat_history.append(HumanMessage(content=msg["content"]))
                else:
                    chat_history.append(AIMessage(content=msg["content"]))
        
        return chat_history
    


