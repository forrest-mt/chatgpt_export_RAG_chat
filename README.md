# ChatGPT History Chat

A personal AI-powered chat application that lets you have conversations with your own ChatGPT history data using RAG (Retrieval-Augmented Generation).

## What it does

This app allows you to:
- Upload your ChatGPT conversation history 
- Chat with an AI that has context from your past conversations
- Keep your personal data private (runs locally with Ollama)
- Use your chat history as a "second brain" for recalling past thoughts and conversations

## Features

- **Local AI**: Uses Ollama for privacy-focused local LLM inference
- **RAG System**: Retrieves relevant context from your chat history before responding
- **Vector Search**: ChromaDB for efficient similarity search through your conversations  
- **Web Interface**: Clean Gradio UI for easy chatting
- **Smart Ranking**: Uses cross-encoder reranking for better context relevance

## Tech Stack

- **Frontend**: Gradio
- **LLM**: Ollama (Gemma2 3B)
- **Vector DB**: ChromaDB  
- **Embeddings**: HuggingFace all-MiniLM-L6-v2
- **Framework**: LlamaIndex

## Setup

1. Install dependencies: `uv install`
2. Start Ollama and pull the model: `ollama pull gemma2:3b`
3. Process your ChatGPT data using the Jupyter notebook
4. Run the app: `uv run app.py`

See `setup.md` for detailed setup instructions.