# RAG AI Assistant — Developer Guide

This document explains how this project works from the ground up. No prior knowledge of AI or any of the libraries used is assumed. Read it top-to-bottom before touching any code.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Background Concepts](#2-background-concepts)
   - [What is RAG?](#what-is-rag)
   - [What are Embeddings?](#what-are-embeddings)
   - [What is a Vector Database?](#what-is-a-vector-database)
   - [What is Hybrid Search?](#what-is-hybrid-search)
   - [What is LangChain?](#what-is-langchain)
3. [System Architecture](#3-system-architecture)
4. [The Two-Stage Pipeline (The Brain)](#4-the-two-stage-pipeline-the-brain)
   - [Stage 1 — Query Router & Reformulator](#stage-1--query-router--reformulator)
   - [Stage 2 — Answer Generator](#stage-2--answer-generator)
5. [File-by-File Breakdown](#5-file-by-file-breakdown)
6. [Services & Infrastructure](#6-services--infrastructure)
7. [Data Ingestion (One-Time Setup)](#7-data-ingestion-one-time-setup)
8. [Memory System](#8-memory-system)
9. [Environment Variables](#9-environment-variables)
10. [Setup & Installation](#10-setup--installation)
    - [Prerequisites](#prerequisites)
    - [Step 1 — Get a Groq API Key](#step-1--get-a-groq-api-key)
    - [Step 2 — Clone the Repository](#step-2--clone-the-repository)
    - [Step 3 — Create a Virtual Environment](#step-3--create-a-virtual-environment)
    - [Step 4 — Install Python Dependencies](#step-4--install-python-dependencies)
    - [Step 5 — Create the .env File](#step-5--create-the-env-file)
    - [Step 6 — Start the Docker Services](#step-6--start-the-docker-services)
    - [Step 7 — Ingest the Data](#step-7--ingest-the-data)
    - [Step 8 — Run the Project](#step-8--run-the-project)
    - [Stopping the Project](#stopping-the-project)
    - [Troubleshooting](#troubleshooting)
11. [Your Job — Replacing the Discord Interface](#11-your-job--replacing-the-discord-interface)
12. [The Public API You Need to Call](#12-the-public-api-you-need-to-call)

---

## 1. What This Project Does

This is an AI chatbot that answers questions about an internal DevOps knowledge base. Instead of a generic chatbot that only knows what it was trained on, this bot can look things up in a private database of documents — incident logs, infrastructure maps, credentials, cheatsheets — and give accurate, grounded answers.


---

## 2. Background Concepts

Before understanding the code, you need to understand four concepts.

### What is RAG?

**RAG** stands for **Retrieval-Augmented Generation**. It is a pattern for making AI smarter by giving it access to a private database of documents at query time.

Here is the problem it solves: a plain AI model like GPT or Llama only knows what it was trained on. If you ask it "What is the root cause of Incident #9902?", it has no idea — that incident is internal, private data that was never in its training set.

RAG solves this by splitting the work into two steps:
1. **Retrieve** — Before answering, search a private database for documents relevant to the question.
2. **Generate** — Feed those documents to the AI as extra context, then ask it to answer.

The AI is no longer guessing from memory. It is reading the relevant documents and synthesizing an answer from them, just like a human employee would look something up before answering.

```
User Question
     │
     ▼
┌─────────────┐        ┌─────────────────────┐
│  Search the │───────▶│  Relevant Documents │
│  Database   │        │  (from your files)  │
└─────────────┘        └──────────┬──────────┘
                                  │
                                  ▼
                        ┌─────────────────────┐
                        │  AI reads the docs  │
                        │  and answers the    │
                        │  question           │
                        └─────────────────────┘
```

### What are Embeddings?

A plain text search (like `Ctrl+F`) only finds exact keyword matches. If you search for "server ran out of memory", it will not find a document that says "OOM Killer was invoked" — even though they mean the same thing.

**Embeddings** solve this. An embedding is a way to convert a piece of text into a list of numbers (a vector) that captures its *meaning*. Two sentences with similar meaning will produce vectors that are numerically close to each other, even if they use completely different words.

```
"The server ran out of memory"  →  [0.12, -0.83, 0.41, 0.07, ...]  ──┐
                                                                       │ (these are close)
"OOM Killer was invoked"        →  [0.15, -0.79, 0.38, 0.10, ...]  ──┘

"My cat is fluffy"              →  [-0.65, 0.22, -0.91, 0.55, ...] (far from both above)
```

This project uses a model called **`BAAI/bge-small-en-v1.5`** (via a library called FastEmbed) to convert text into 384-dimensional vectors. This model runs entirely locally — no external API calls are needed for embeddings.

### What is a Vector Database?

A **vector database** is a database designed specifically to store and search these number vectors. When you give it a query vector, it finds the stored vectors that are most numerically similar to it (i.e., the most semantically similar text chunks).

This project uses **Qdrant** as its vector database. Qdrant runs as a Docker container on your machine and stores its data in the `qdrant_data/` folder.

Think of it this way:
- A regular SQL database stores rows and lets you search by exact field values.
- A vector database stores "meaning fingerprints" and lets you search by semantic similarity.

### What is Hybrid Search?

This project uses **Hybrid Search**, which combines two types of search simultaneously:

| Search Type | How It Works | Good At |
|---|---|---|
| **Dense (Semantic)** | Uses the embedding vectors described above. Finds conceptually similar text. | "What went wrong with the database?" → finds OOM Killer incident |
| **Sparse (Keyword/BM25)** | Traditional keyword frequency matching (like a search engine). Finds exact terms. | "INCIDENT #9902" → finds the exact incident log |

By combining both, the system is good at both fuzzy concept searches and precise keyword searches. The sparse embedding model used is **`Qdrant/bm25`**.

### What is LangChain?

**LangChain** is a Python framework for building applications powered by language models. Think of it as the "plumbing" that connects all the pieces together:
- It provides a standard interface to talk to different AI models (Groq, OpenAI, etc.)
- It has pre-built components for creating chains of operations (e.g., "send to LLM, then parse output, then query database")
- It handles prompt templates, message history formatting, and retrieval logic

This project uses LangChain to wire together the Groq AI API, the Qdrant database, and the prompt templates.

---

## 3. System Architecture

The project is made of four independent services that talk to each other:

```
┌──────────────────────────────────────────────────────────┐
│                     Your Machine                         │
│                                                          │
│  ┌────────────────┐      ┌─────────────────────────┐    │
│  │  Python Bot    │      │  Docker Containers       │    │
│  │  (main.py)     │      │                         │    │
│  │                │      │  ┌─────────────────────┐│    │
│  │  ┌──────────┐  │      │  │  Qdrant Vector DB   ││    │
│  │  │RagEngine │◀─┼──────┼─▶│  Port: 6333         ││    │
│  │  │(rag_logic│  │      │  │  (stores embeddings)││    │
│  │  │   .py)   │  │      │  └─────────────────────┘│    │
│  │  └──────────┘  │      │                         │    │
│  │                │      │  ┌─────────────────────┐│    │
│  │  ┌──────────┐  │      │  │  Redis Cache        ││    │
│  │  │  Redis   │◀─┼──────┼─▶│  Port: 6379         ││    │
│  │  │  Memory  │  │      │  │  (stores chat logs) ││    │
│  │  └──────────┘  │      │  └─────────────────────┘│    │
│  └───────┬────────┘      └─────────────────────────┘    │
│          │                                               │
└──────────┼───────────────────────────────────────────────┘
           │
           ▼ (External)
     ┌───────────┐
     │ Groq API  │
     │ (LLM)     │
     │ [Internet]│
     └───────────┘
```

- **Python Bot** — The main application. Receives user messages, runs the AI pipeline, returns answers.
- **Qdrant** — The vector database. Stores the knowledge base and answers search queries.
- **Redis** — An in-memory cache. Stores each user's conversation history temporarily (expires after 5 minutes).
- **Groq API** — A cloud service that runs a Llama 3.1 language model. This is the "thinking" part. Requires an internet connection and an API key.

---

## 4. The Two-Stage Pipeline (The Brain)

This is the most important part to understand. A single user message does **not** go straight to the AI for an answer. It goes through **two separate LLM calls** in sequence. This is a deliberate design choice to save cost and improve quality.

```
User Message + Chat History
         │
         ▼
┌──────────────────────────────────┐
│  STAGE 1: Query Router           │
│  (LLM Call #1 — cheap/fast)      │
│                                  │
│  • Reads the message + history   │
│  • Rewrites pronouns to names    │
│  • Decides: does this need RAG?  │
│  • If YES: outputs search terms  │
│  • If NO: outputs "NO_RAG"       │
└─────────────────┬────────────────┘
                  │
        ┌─────────┴─────────┐
        │ Output contains   │
        │ "NO_RAG"?         │
        └─────┬─────────────┘
              │
        ┌─────▼──────────────────────────────────┐
        │ YES: "No Context,                       │
        │       answer from General Knowledge"    │
        └─────┬───────────────────────────────────┘
              │                  │ NO: Run search terms
              │                  ▼
              │         ┌───────────────────┐
              │         │  Qdrant Hybrid     │
              │         │  Search (top 3)    │
              │         │  Returns text      │
              │         │  chunks            │
              │         └────────┬──────────┘
              │                  │
              ▼                  ▼
        ┌──────────────────────────────────────────┐
        │  STAGE 2: Answer Generator               │
        │  (LLM Call #2)                           │
        │                                          │
        │  System Prompt = SYSTEM_PROMPT2          │
        │                + Retrieved Context       │
        │                                          │
        │  Input = Chat History + User Question    │
        │                                          │
        │  Output = Final answer sent to user      │
        └──────────────────────────────────────────┘
```

### Stage 1 — Query Router & Reformulator

**File:** `bot/config.py` (contains `SYSTEM_PROMPT1`)
**File:** `bot/rag_logic.py` (method `get_rag_response`, first half)

The first LLM call uses `SYSTEM_PROMPT1`, which turns the AI into a **Query Reformulator**. Its job is not to answer — it is to prepare the query for searching.

It does three things:

**1. Context Resolution (replaces pronouns)**

If a user says "How do I fix it?" after a conversation about a server crash, this is ambiguous. The router uses the chat history to rewrite this as something specific like "How do I fix the OOM Killer crash on db-prod-primary?" This makes the database search much more effective.

**2. Topic Filtering**

If the user says "Hello!" or "What's the weather?", there is no point searching the DevOps database. The router outputs the special token `NO_RAG` to signal that no database lookup is needed, and the second stage will answer from the AI's general knowledge.

**3. Query Extraction**

For technical questions, the router extracts one or more specific, standalone search queries separated by newlines. These queries are designed to be optimal for searching the database.

**Example:**

```
Chat History: "User: Incident #9902, AI: What would you like to know?"
User Input: "What was the fix?"

Stage 1 Output:
  "What was the resolution for Incident #9902?"
  "How was the OOM Killer crash resolved on db-prod-primary?"
```

These two queries are then used to search Qdrant in parallel, and the top 3 matching document chunks from each query are collected and merged.

### Stage 2 — Answer Generator

**File:** `bot/config.py` (contains `SYSTEM_PROMPT2`)
**File:** `bot/rag_logic.py` (method `get_rag_response`, second half)

The second LLM call uses `SYSTEM_PROMPT2`. By this point, the system has already retrieved relevant document chunks from the database.

The final prompt passed to the LLM looks like this:

```
[SYSTEM_PROMPT2 text]

[DATABASE CONTEXT]:
--- Content of retrieved chunk 1 ---
--- Content of retrieved chunk 2 ---
--- Content of retrieved chunk 3 ---

[Chat History as conversation turns]

[User Question]
```

The LLM reads all of this and generates a final answer. Because the relevant documents are directly in the prompt, the AI can cite specific details (IPs, passwords, commands) accurately without hallucinating them.

---

## 5. File-by-File Breakdown

```
AI_Project/
│
├── bot/
│   ├── main.py          ← Entry point. Runs the  CLI interface.
│   │                      YOUR TEAMMATES REPLACE THIS FILE.
│   │
│   ├── rag_logic.py     ← The core AI engine. Contains the RagEngine class
│   │                      and the two-stage pipeline. DO NOT MODIFY.
│   │
│   ├── config.py        ← The two system prompts (SYSTEM_PROMPT1, SYSTEM_PROMPT2).
│   │                      The "personality" and "instructions" for the AI.
│   │
│   ├── memory.py        ← The RedisMemory class. Handles saving and loading
│   │                      per-user chat history from Redis.
│   │
│   ├── ingest.py        ← One-time data loading script. Reads files from Data/,
│   │                      converts them to embeddings, stores in Qdrant.
│   │                      Run ONCE before starting the bot.
│   │
│   └── requirements.txt ← All Python package dependencies.
│
├── Data/
│   ├── dev_cheatsheet.txt    ← Internal developer guide (commands, workflows)
│   ├── incident_logs.txt     ← History of system incidents and resolutions
│   ├── infrastructure_map.txt← Server IPs, environments, connection strings
│   └── secrets_vault.txt     ← Internal credentials and keys
│
├── qdrant_data/         ← Qdrant's persistent storage. Auto-managed by Docker.
│                          Do not manually edit files here.
│
├── docker-compose.yml   ← Defines the Qdrant and Redis services.
│
└── README.md            ← This file.
```

---

## 6. Services & Infrastructure

The project requires two background services that run in Docker containers. They are defined in `docker-compose.yml`.

### Qdrant (Vector Database)

- **What it is:** The searchable knowledge base. Stores all the document chunks as mathematical vectors.
- **Port:** `6333`
- **Data:** Persisted to `./qdrant_data` on your disk.
- **Web UI:** You can inspect the database at `http://localhost:6333/dashboard` once it is running.

### Redis (Cache)

- **What it is:** An extremely fast in-memory key-value store. Used here as a temporary conversation memory.
- **Port:** `6379`
- **How it stores data:** Each user has a Redis key (their user ID). The value is a list of JSON-encoded messages. Each list item is one turn of conversation.
- **TTL (Time-To-Live):** Each key expires after **300 seconds (5 minutes)** of inactivity. After that, the bot forgets the conversation.
- **Message limit:** Maximum 20 messages (10 turns) are stored per user. Older messages are dropped as new ones arrive.

---

## 7. Data Ingestion (One-Time Setup)

Before the bot can answer any questions, the documents in the `Data/` folder must be processed and loaded into Qdrant. This is done by running `ingest.py` **once**.

`ingest.py` performs the following pipeline:

```
Data/*.txt files
      │
      ▼
DirectoryLoader (LangChain)
  Reads all .txt files into memory
      │
      ▼
RecursiveCharacterTextSplitter
  Breaks each document into chunks of ~350 characters
  with a 50-character overlap between adjacent chunks.
  (Overlap ensures context is not lost at chunk boundaries)
      │
      ▼
FastEmbedEmbeddings (BAAI/bge-small-en-v1.5)
  Converts each chunk into a 384-dimensional dense vector.
  Runs locally on CPU. No API needed.
      │
      ▼
FastEmbedSparse (Qdrant/bm25)
  Converts each chunk into a sparse keyword vector.
  This enables keyword-based search.
      │
      ▼
QdrantVectorStore
  Creates a collection named "knowledge_base" in Qdrant.
  Uploads all chunks with BOTH their dense and sparse vectors.
  The data is now permanently stored in qdrant_data/ on disk.
```

**To re-ingest data** (e.g., after modifying files in `Data/`): run `ingest.py` again. It deletes the old collection and rebuilds it from scratch.

---

## 8. Memory System

The `memory.py` file contains the `RedisMemory` class. It manages per-user conversation history.

**How it works:**

Each user is identified by a string key (in the Discord bot, this is their Discord user ID). Conversation turns are stored as a Redis list under that key.

```python
# Adding a message
memory.add_message("user_12345", HumanMessage(content="What is Incident #9902?"))
memory.add_message("user_12345", AIMessage(content="Incident #9902 was an OOM crash..."))

# Retrieving all messages for a user
history = memory.get_messages("user_12345")
# Returns: [HumanMessage(...), AIMessage(...), ...]
```

The returned list is a sequence of `HumanMessage` and `AIMessage` objects from LangChain. This is the format that `get_rag_response()` expects as its `chat_history` parameter.

**Why Redis and not a simple Python list?**

A plain Python list would lose its data every time the program restarts, and it cannot be shared between multiple processes. Redis is persistent across restarts and can be accessed from any process on the machine.

---

## 9. Environment Variables

The bot reads secrets from a `.env` file in the `bot/` directory. This file is **not committed to git** (it contains real keys). You must create it yourself.

Create `bot/.env` with the following contents:

```env
# Groq API key — get one free at https://console.groq.com
Ai_Api_Key=your_groq_api_key_here

```

---

## 10. Setup & Installation

Follow these steps in order. Do not skip any — each step depends on the previous one.

### Prerequisites

Before you start, make sure you have the following installed on your machine.

**Python 3.10 or newer**

Check your version:
```bash
python --version
# or on some systems:
python3 --version
```

If you do not have Python 3.10+, download it from [python.org](https://www.python.org/downloads/).

**Docker Desktop**

Docker runs the Qdrant and Redis services as containers so you do not need to install them manually.

- **Windows / macOS:** Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/).
- **Linux:** Install Docker Engine and the Compose plugin:
  ```bash
  sudo apt-get install docker.io docker-compose-v2   # Ubuntu/Debian
  sudo dnf install docker docker-compose-plugin      # Fedora
  ```

After installing, verify Docker is running:
```bash
docker --version
docker compose version
```

> **Note:** Older Docker installations use `docker-compose` (with a hyphen) as a separate command. Newer ones use `docker compose` (as a subcommand). Both work. The examples below use `docker compose`.

---

### Step 1 — Get a Groq API Key

The project uses Groq to run the Llama 3.1 language model. Groq offers a **free tier** that is more than sufficient for this project.

1. Go to [console.groq.com](https://console.groq.com) and create a free account.
2. Navigate to **API Keys** in the left sidebar.
3. Click **Create API Key**, give it a name (e.g. `ai-project`), and copy the key.
4. Save the key somewhere safe — you will need it in Step 5.

---

### Step 2 — Clone the Repository

```bash
git clone <repository-url>
cd AI_Project
```

---

### Step 3 — Create a Virtual Environment

A virtual environment keeps this project's Python packages isolated from the rest of your system. This prevents version conflicts with other projects.

```bash
# From the AI_Project/ root directory
python -m venv .venv
```

Then activate it:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (Command Prompt)
.venv\Scripts\activate.bat

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

Your terminal prompt will change to show `(.venv)` at the start, confirming the environment is active. **You must activate the virtual environment every time you open a new terminal.**

---

### Step 4 — Install Python Dependencies

With the virtual environment active, install all required packages:

```bash
cd bot/
pip install -r requirements.txt
```

This will download and install everything listed in `requirements.txt`. It may take a few minutes the first time because the FastEmbed library downloads AI model files (~100 MB).

Verify the install succeeded:
```bash
pip list | grep langchain
# Should print several langchain-* packages
```

---

### Step 5 — Create the .env File

The project needs secret keys that must not be committed to git. Create the file `bot/.env` (note: this file is already in `.gitignore`):

```bash
# From the bot/ directory
touch .env        # macOS / Linux
# On Windows, just create a new file called ".env" in the bot/ folder
```

Open `bot/.env` in any text editor and add the following, replacing the placeholder with your actual Groq key from Step 1:

```env
Ai_Api_Key=your_groq_api_key_here
```

> `Bot_Token` is only needed for the Discord interface, which is being replaced. Leave it out entirely.

Verify the file looks correct:
```bash
cat .env
# Should print: Ai_Api_Key=gsk_...
```

---

### Step 6 — Start the Docker Services

Go back to the project root and start Qdrant and Redis:

```bash
# From the AI_Project/ root directory
docker compose up -d
```

The first run will download the Docker images (~200 MB total). Subsequent starts are instant.

Verify both containers are running:
```bash
docker compose ps
```

You should see two containers with `Status: running` (or `Up`):

```
NAME          STATUS
rag-qdrant    running
rag-redis     running
```

You can also open the Qdrant web dashboard in your browser to confirm it is live:
```
http://localhost:6333/dashboard
```

> **Important:** The Docker services must be running before you can run `ingest.py` or start the bot. You will need to run `docker compose up -d` again after rebooting your machine.

---

### Step 7 — Ingest the Data

This step reads all the `.txt` files from the `Data/` folder, converts them into embeddings, and loads them into Qdrant. **You only need to do this once** (or again if you modify the files in `Data/`).

```bash
# From the bot/ directory, with the virtual environment active
python ingest.py
```

Expected output:
```
Sparse Embeddings is <class '...FastEmbedSparse'>
Success. Data was inserted successfully.
```

Confirm the data is in Qdrant by opening `http://localhost:6333/dashboard` and clicking on the `knowledge_base` collection. You should see several hundred points (document chunks).

> **If ingest.py fails:** Make sure the Docker containers from Step 6 are running (`docker compose ps`). The script connects to Qdrant on `localhost:6333` — if Qdrant is not running, the connection will be refused.

---

### Step 8 — Run the Project

```bash
# From the bot/ directory, with the virtual environment active
python main.py
```

The original `main.py` runs a CLI interface.

To test that the AI engine is working before building your interface, you can temporarily run `main.py` and type questions in the CLI.

---

### Stopping the Project

To stop the Docker containers when you are done:

```bash
# From the AI_Project/ root directory
docker compose down
```

This stops and removes the containers but **preserves the data** in `qdrant_data/`. You will not need to re-run `ingest.py` next time.

To stop the containers without removing them (faster restart):
```bash
docker compose stop
```

---

### Troubleshooting

**`ModuleNotFoundError` when running Python scripts**

The virtual environment is not active. Run `source .venv/bin/activate` (macOS/Linux) or `.venv\Scripts\activate.bat` (Windows) first.

**`Connection refused` to localhost:6333 or localhost:6379**

The Docker containers are not running. From the project root:
```bash
docker compose up -d
docker compose ps   # confirm both show "running"
```

**`AuthenticationError` or `401` from Groq**

Your API key in `bot/.env` is missing or incorrect. Double-check the file contains `Ai_Api_Key=gsk_...` with your actual key.

**`ingest.py` runs but shows 0 points in Qdrant dashboard**

Make sure you are running `ingest.py` from inside the `bot/` directory, not the project root. The script looks for a `Data/` folder relative to its location:
```bash
cd bot/
python ingest.py
```

**`docker compose` not found (older Docker installations)**

Try `docker-compose` with a hyphen instead:
```bash
docker-compose up -d
docker-compose ps
```

--

## 12. The Public API You Need to Call

Everything you need is exposed through two classes: `RagEngine` and `RedisMemory`.

### Initialization

```python
from rag_logic import RagEngine
from memory import RedisMemory
from langchain_core.messages import HumanMessage, AIMessage

# Create once at startup — this loads models and connects to Qdrant
engine = RagEngine()

# Create a memory handler
memory = RedisMemory()
```

`RagEngine()` takes a few seconds to initialize because it loads the embedding models. Create it **once** when your application starts, not once per message.

### Sending a Message and Getting a Response

```python
# A unique string that identifies this user (could be a username, UUID, etc.)
user_id = "some_unique_user_identifier"

# 1. Load conversation history (returns [] if user has no history)
history = memory.get_messages(user_id)

# 2. Get the AI's response
# - question: the user's message as a plain string
# - history: the list returned by get_messages()
response_text = engine.get_rag_response(
    question="What was the fix for Incident #9902?",
    chat_history=history
)

# response_text is a plain string — the AI's answer

# 3. Save the new exchange to memory so future messages have context
memory.add_message(user_id, HumanMessage(content="What was the fix for Incident #9902?"))
memory.add_message(user_id, AIMessage(content=response_text))
```

### Clearing a User's History

To implement a "reset" or "new conversation" button, simply delete the user's key from Redis:

```python
memory.client.delete(user_id)
```

### Important: Threading

`engine.get_rag_response()` is a **synchronous, blocking** function. It makes two network calls to the Groq API and one call to Qdrant. In a GUI or web app with an event loop (e.g., asyncio, tkinter, Qt), you must run it in a background thread to avoid freezing the UI.

Python example using `asyncio`:

```python
import asyncio

loop = asyncio.get_running_loop()
response_text = await loop.run_in_executor(
    None,  # uses the default thread pool
    lambda: engine.get_rag_response(user_question, history)
)
```



## Summary of Data Flow

Every time a user sends a message, this is the complete path through the system:

```
User Input
    │
    ▼
RedisMemory.get_messages(user_id)
    │  Returns list of past HumanMessage / AIMessage objects
    ▼
RagEngine.get_rag_response(question, chat_history)
    │
    ├─▶ [LLM Call 1 — Groq API]
    │       SYSTEM_PROMPT1 (Router)
    │       Input: question + chat history
    │       Output: reformulated search terms, OR "NO_RAG"
    │
    ├─▶ [Qdrant Hybrid Search]  ← skipped if output was "NO_RAG"
    │       Dense search (semantic similarity via BGE embeddings)
    │       Sparse search (keyword match via BM25)
    │       Returns top 3 most relevant text chunks
    │
    └─▶ [LLM Call 2 — Groq API]
            SYSTEM_PROMPT2 (Answer Generator)
            Input: question + chat history + retrieved chunks
            Output: final answer string
    │
    ▼
RedisMemory.add_message(user_id, HumanMessage)
RedisMemory.add_message(user_id, AIMessage)
    │  Saves this exchange for future context (TTL: 5 minutes)
    ▼
Return response string to caller
    │
    ▼
Your UI displays it to the user
```
