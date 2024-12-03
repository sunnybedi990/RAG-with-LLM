from rag_models import RAG_with_groq
from rag_models import Rag_with_ollama
from rag_models import RAG_with_openai

def generate_response(prompt, model, provider):
    if provider.lower() == "groq":
        return RAG_with_groq(prompt, model)
    elif provider.lower() == "ollama":
        return Rag_with_ollama(prompt, model)
    elif provider.lower() == "openai":
        return RAG_with_openai(prompt, model)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
