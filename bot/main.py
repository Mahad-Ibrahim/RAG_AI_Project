import discord
import os
import asyncio
import threading
import sys
from dotenv import load_dotenv
from rag_logic import RagEngine
from memory import RedisMemory
from langchain_core.messages import HumanMessage, AIMessage

# --- ANSI COLORS FOR CLI ---
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

load_dotenv()
Bot_Token = os.getenv("Bot_Token")

print(f"{Colors.HEADER}--- SYSTEM STARTUP ---{Colors.ENDC}")
print(f"{Colors.BOLD}Initializing Logic-Router Engine...{Colors.ENDC}")
try:
    shared_rag_engine = RagEngine()
    print(f"{Colors.GREEN}[SUCCESS] RAG Engine Online.{Colors.ENDC}")
except Exception as e:
    print(f"{Colors.FAIL}[CRITICAL] Engine failed: {e}{Colors.ENDC}")
    sys.exit(1)

intents = discord.Intents.default()
intents.message_content = True 

class DevOpsBot(discord.Client):
    def __init__(self, engine):
        super().__init__(intents=intents)
        self.rag_engine = engine # Use the shared engine
        self.memory = RedisMemory()
        
    async def on_ready(self):
        print(f"\n{Colors.WARNING}[DISCORD]{Colors.ENDC} Logged in as {self.user} (ID: {self.user.id})")
        print(f"{Colors.WARNING}[DISCORD]{Colors.ENDC} Ready for switching in video...\n")
        # Reprint CLI prompt after async interruption
        print(f"{Colors.GREEN}You (CLI):{Colors.ENDC} ", end="", flush=True)

    async def on_message(self, message):
        if message.author.bot:
            return

        print(f"\n{Colors.WARNING}[DISCORD MSG]{Colors.ENDC} User {message.author.name}: {message.content}")
        print(f"{Colors.GREEN}You (CLI):{Colors.ENDC} ", end="", flush=True) # Keep CLI prompt clean

        user_id = str(message.author.id) 

        if message.content == "!reset":
            embed = discord.Embed(title="♻️ Memory Wiped", description="History cleared.", color=discord.Color.red())
            await message.channel.send(embed=embed)
            return

        async with message.channel.typing():
            try:
                history_objects = self.memory.get_messages(user_id)
                
                # Run RAG in non-blocking way
                loop = asyncio.get_running_loop()
                response_text = await loop.run_in_executor(
                    None, 
                    lambda: self.rag_engine.get_rag_response(message.content, history_objects)
                )

                # Save Memory
                self.memory.add_message(user_id, HumanMessage(content=message.content))
                self.memory.add_message(user_id, AIMessage(content=response_text))

                # Send Rich Embed
                await self.send_long_embed(message.channel, response_text, message)

            except Exception as e:
                print(f"{Colors.FAIL}[DISCORD ERROR] {e}{Colors.ENDC}")
                await message.reply(f"System Error: {e}")

    async def send_long_embed(self, channel, text, original_message):
        chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
        for i, chunk in enumerate(chunks):
            embed = discord.Embed(
                description=chunk,
                color=discord.Color.from_rgb(43, 45, 49) # Dark Code Gray
            )
            if i == 0:
                embed.set_author(name="DevOps Assistant", icon_url=self.user.avatar.url if self.user.avatar else None)
                embed.title = "🔎 Analysis Result"
            if i == len(chunks) - 1:
                embed.set_footer(text=f"Logic-Router Engine | Context-Aware | User: {original_message.author.name}")

            await channel.send(embed=embed)

def start_cli_interface(engine):
    cli_user_id = "cli_admin_user"
    local_memory = RedisMemory() 
    
    print(f"{Colors.BOLD}--- VIDEO RECORDING SESSION STARTED ---{Colors.ENDC}")
    print("Type in CLI to chat. Alt-Tab to Discord to test integration.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input(f"{Colors.GREEN}You (CLI):{Colors.ENDC} ")
            
            if user_input.lower() in ["exit", "quit"]:
                print(f"{Colors.BLUE}[INFO]{Colors.ENDC} Shutting down...")
                os._exit(0) # Force kill threads

            if not user_input.strip():
                continue

            print(f"{Colors.WARNING}Thinking...{Colors.ENDC}", end="\r")
            
            # Fetch History
            history = local_memory.get_messages(cli_user_id)
            
            # Get Response
            response = engine.get_rag_response(user_input, history)
            
            # Save History
            local_memory.add_message(cli_user_id, HumanMessage(content=user_input))
            local_memory.add_message(cli_user_id, AIMessage(content=response))
            
            # Clear Loading
            print(" " * 20, end="\r")
            
            # Print Output
            print(f"{Colors.BLUE}Bot (CLI):{Colors.ENDC} {response}\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n{Colors.FAIL}[CLI ERROR] {e}{Colors.ENDC}")

if __name__ == "__main__":
    if not Bot_Token:
        print("Error: Bot_Token not found in .env")
        sys.exit(1)

    bot = DevOpsBot(shared_rag_engine)

    discord_thread = threading.Thread(target=bot.run, args=(Bot_Token,), daemon=True)
    discord_thread.start()
    
    start_cli_interface(shared_rag_engine)

