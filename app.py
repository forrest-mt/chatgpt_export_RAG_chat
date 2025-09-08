import gradio as gr
import chromadb
from llama_index.core import Settings, PromptTemplate, VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import CrossEncoder

## Constants
RETRIEVE_K = 12
TOP_N = 5
MODEL = "gemma3:1b"  # or "gemma3:4b"
TEMPERATURE = 0.1
EMBED_MODEL = "all-MiniLM-L6-v2"

# Lazy init for CrossEncoder to avoid slow startup
_ce = None
def get_ce():
    global _ce
    if _ce is None:
        _ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _ce

# Use local embeddings instead of OpenAI
Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")

# Your existing LLM config
from llama_index.llms.ollama import Ollama
Settings.llm = Ollama(model="gemma3:1b", request_timeout=60.0, temperature=0.1)
ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

## Prompt Template
qa_tmpl = PromptTemplate(
    "You are a concise, grounded assistant.\n"
    "Only use the provided context. If an answer is not supported by the context, say you don't know.\n"
    "Do not include any external links or sources; cite only the provided source titles.\n"
    "Return a short synthesis.\n\n"
    "Context:\n{context_str}\n\n"
    "Question: {query_str}\n\n"
    "Answer:"
)

# 1) Load Chroma + index once
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="chat_history")
vector_store = ChromaVectorStore(chroma_collection=collection)
index = VectorStoreIndex.from_vector_store(vector_store)

def rag_query(q: str, retrieve_k=10, top_n=5):
    retriever = index.as_retriever(similarity_top_k=retrieve_k)
    nodes = retriever.retrieve(q)
    pairs = [(q, n.text or "") for n in nodes]
    scores = ce.predict(pairs)

    ranked = sorted(zip(nodes, scores), key=lambda x: x[1], reverse=True)[:top_n]
    top_nodes = [n for n, _ in ranked]
    ctx = "\n\n".join((n.text or "") for n in top_nodes)

    prompt = qa_tmpl.format(context_str=ctx, query_str=q)
    answer = Settings.llm.complete(prompt).text

    sources = []
    for n, s in ranked:
        md = n.metadata or {}
        sources.append({
            "score": float(s),
            "title": md.get("source_name", "Unknown"),
            "author": md.get("author", "Unknown"),
            "preview": (n.text or "").replace("\n", " ")[:180]
        })
    return answer, sources

def ui_fn(query):
    if not query.strip():
        return "Please enter a journal entry or question.", ""
    answer, sources = rag_query(query, retrieve_k=12, top_n=5)
    src_text = "\n".join(
        [f"{i+1}. score={s['score']:.3f} | {s['author']} — {s['title']}\n   {s['preview']}..."
         for i, s in enumerate(sources)]
    )
    return answer, src_text

def chat_fn(message, history, rag_context=""):
    """Handle chat conversation with optional RAG context"""
    if not message.strip():
        return history, ""
    
    # Build context from conversation history and RAG if available
    context_parts = []
    
    if rag_context.strip():
        context_parts.append(f"Background context from your journal:\n{rag_context}")
    
    # Add recent conversation history (last 3 exchanges)
    if history:
        recent_history = history[-6:]  # Last 3 user-assistant pairs
        context_parts.append("Recent conversation:")
        for turn in recent_history:
            context_parts.append(f"You: {turn[0]}")
            context_parts.append(f"Assistant: {turn[1]}")
    
    # Create prompt for conversational response
    full_context = "\n\n".join(context_parts) if context_parts else ""
    
    if full_context:
        chat_prompt = (
            f"You are a thoughtful conversation partner helping with personal reflection.\n"
            f"Use the provided context to inform your response, but respond conversationally.\n\n"
            f"Context:\n{full_context}\n\n"
            f"Current message: {message}\n\n"
            f"Response:"
        )
    else:
        chat_prompt = (
            f"You are a thoughtful conversation partner helping with personal reflection.\n"
            f"Respond conversationally to: {message}"
        )
    
    # Get response from LLM
    response = Settings.llm.complete(chat_prompt).text.strip()
    
    # Update history
    history.append((message, response))
    
    return history, ""

def use_rag_in_chat(rag_answer, rag_sources):
    """Transfer RAG results to chat context"""
    if not rag_answer.strip():
        return "No RAG context to transfer."
    
    context_text = f"RAG Answer: {rag_answer}\n\nSources: {rag_sources}"
    return context_text

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 ChatGPT History Analysis")
    gr.Markdown("Search your ChatGPT conversations on the left, then continue the conversation on the right.")
    
    with gr.Row():
        # Left Column - RAG Search
        with gr.Column(scale=1):
            gr.Markdown("## 🔍 Conversation Search")
            rag_input = gr.Textbox(
                label="Search your conversations", 
                placeholder="What would you like to explore from your past conversations?",
                lines=4
            )
            search_btn = gr.Button("🔍 Search", variant="primary")
            
            rag_answer = gr.Textbox(label="📝 Synthesized Answer", lines=6)
            rag_sources = gr.Textbox(label="📚 Sources", lines=8)
            
            transfer_btn = gr.Button("➡️ Use in Chat", variant="secondary")
        
        # Right Column - Chat Interface  
        with gr.Column(scale=1):
            gr.Markdown("## 💬 Continue the Conversation")
            
            chat_context = gr.Textbox(
                label="🎯 Chat Context (from RAG)", 
                placeholder="Context will appear here when you transfer RAG results...",
                lines=4,
                visible=True
            )
            
            chatbot = gr.Chatbot(
                type='messages',
                label="Conversation",
                height=400,
                show_label=True
            )
            
            chat_input = gr.Textbox(
                label="Your message",
                placeholder="Continue the conversation based on conversation insights...",
                lines=2
            )
            
            with gr.Row():
                chat_btn = gr.Button("💬 Send", variant="primary")
                clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
    
    # Event handlers
    search_btn.click(
        ui_fn, 
        inputs=[rag_input], 
        outputs=[rag_answer, rag_sources]
    )
    
    transfer_btn.click(
        use_rag_in_chat,
        inputs=[rag_answer, rag_sources],
        outputs=[chat_context]
    )
    
    chat_btn.click(
        chat_fn,
        inputs=[chat_input, chatbot, chat_context],
        outputs=[chatbot, chat_input]
    )
    
    chat_input.submit(  # Allow Enter key to send
        chat_fn,
        inputs=[chat_input, chatbot, chat_context],
        outputs=[chatbot, chat_input]
    )
    
    clear_btn.click(
        lambda: ([], ""),
        outputs=[chatbot, chat_context]
    )

