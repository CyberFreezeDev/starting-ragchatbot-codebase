# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Management
Use **pip** and **venv** — do not use `uv`.

```bash
# One-time setup
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash
# or: venv\Scripts\activate    # Windows CMD/PowerShell

pip install openai chromadb sentence-transformers fastapi uvicorn python-multipart python-dotenv
```

## Commands

**Run the app** (from repo root, venv activated):
```bash
cd backend && uvicorn app:app --reload --port 8000
```
App at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

There are no tests or linting scripts configured in this project.

## Architecture

This is a RAG (Retrieval-Augmented Generation) chatbot with a FastAPI backend and static HTML/JS frontend. It uses a **local Ollama instance** (`127.0.0.1:11434`) as the LLM backend.

**Request flow:**
1. Frontend (`frontend/script.js`) sends `POST /api/query` with `{ query, session_id }`
2. `backend/app.py` routes to `RAGSystem.query()`
3. `RAGSystem` (`backend/rag_system.py`) is the central orchestrator — it calls `AIGenerator` with a `search_course_content` tool
4. The LLM decides whether to call the tool; if it does, `CourseSearchTool` (`backend/search_tools.py`) queries ChromaDB via `VectorStore` (`backend/vector_store.py`)
5. The LLM synthesizes a final answer from retrieved chunks; the exchange is saved to `SessionManager`

**Tool-use pattern:** The AI is given one tool (`search_course_content`) and uses it at most once per query. After tool execution, a second API call is made without tools to generate the final answer. See `AIGenerator._handle_tool_execution()` in `backend/ai_generator.py`. Tool definitions use the OpenAI format (`type: "function"`, `parameters` key).

**Vector store:** ChromaDB with two collections:
- `course_catalog` — one entry per course (title, instructor, link, lessons metadata as JSON)
- `course_content` — chunked lesson text, filtered by `course_title` and `lesson_number`

Course name matching is fuzzy/semantic: the catalog is queried by vector similarity to resolve partial names before filtering content.

**Document ingestion:** On startup, `app.py` loads all `.txt`/`.pdf`/`.docx` files from `../docs/`. `DocumentProcessor` (`backend/document_processor.py`) parses a structured format:
```
Course Title: ...
Course Link: ...
Course Instructor: ...
Lesson 1: Title
Lesson Link: ...
<content>
```
Chunks are sentence-split with configurable size/overlap (`CHUNK_SIZE=800`, `CHUNK_OVERLAP=100`).

**Key config values** (`backend/config.py`):
- `OLLAMA_BASE_URL`: `http://127.0.0.1:11434/v1`
- `OLLAMA_MODEL`: `qwen2.5`
- `EMBEDDING_MODEL`: `all-MiniLM-L6-v2` (local, via sentence-transformers)
- `MAX_RESULTS`: 5 chunks returned per search
- `MAX_HISTORY`: 2 exchanges (4 messages) kept in session context
- `CHROMA_PATH`: `./chroma_db` (relative to `backend/`)

**Adding a new tool:** Subclass `Tool` (abstract base in `backend/search_tools.py`), implement `get_tool_definition()` and `execute()`, then register with `ToolManager.register_tool()` in `RAGSystem.__init__()`.
