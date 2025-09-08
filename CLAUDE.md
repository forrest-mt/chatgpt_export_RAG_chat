# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Environment

This is a Python project managed with `uv` (Python package manager). The project uses Python 3.9+ and manages dependencies through `pyproject.toml`.

## Instructions
I am a beginner-level programmer. Try to teach me and give me concise and beginner-friendly language. Feel free to use analogies to help me better understand intuitively what is going on. Give me context and understanding.

### Common Dependencies

Based on setup.md, this project is configured to use:
- llama-index (for AI/LLM functionality)
- gradio (for web UI)
- chromadb (for vector database)
- jupyter (for notebooks)
- ollama (for local LLM serving)

## Project Structure

This is a minimal Python project with:
- `main.py`: Entry point with basic "Hello World" functionality
- `pyproject.toml`: Python project configuration and dependencies
- `setup.md`: Development setup instructions
- `prd.md`: Product requirements (currently empty)

## Development Workflow

The project uses `uv` for dependency management. When adding new dependencies, update them in `pyproject.toml` using `uv add <package>`.

The main application entry point is `main()` in `main.py`.