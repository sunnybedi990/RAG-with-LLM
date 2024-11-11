import argparse
from VectorDB import VectorDB
from add_to_vector_db import add_pdf_to_vector_db
from rag_models import RAG_with_groq, Rag_with_ollama, RAG_with_openai  # Import the RAG functions from the other file

def query_vector_db(db_path, query, top_k=5, model="openai",provider='', use_gpu=False):
    """Loads the vector database, performs a search query, and uses the specified model for response generation."""
    db = VectorDB(dimension=384, use_gpu=use_gpu)
    db.load_index(db_path)

    print(f"Querying the vector database with: '{query}'")
    results = db.search(query, top_k=top_k)

    # Combine results into a single prompt
    prompt = f"User's query: {query}\nRelevant information:\n" + "\n".join([text for text, score in results])
    response = ""
    
    if provider.lower() == "groq":
        response = RAG_with_groq(prompt,model)
    elif provider.lower() == "ollama":
        response = Rag_with_ollama(prompt,model)
    elif provider.lower() == "openai":
        response = RAG_with_openai(prompt,model)
    else:
        print(f"Error: Unsupported model '{model}' specified.")
        return
    
    print("\nResponse from the model:")
    print(response)
    return response

# llm_integration.py
def summarize_with_llm(chunks, model, provider):
    """Summarize document chunks using an LLM."""
    # Combine the chunks into a single prompt for the summarization request
    prompt = "Summarize the following text:\n\n" + "\n\n".join([chunk[0] for chunk in chunks])

    if provider.lower() == "groq":
        summary = RAG_with_groq(prompt, model)
    elif provider.lower() == "ollama":
        summary = Rag_with_ollama(prompt, model)
    elif provider.lower() == "openai":
        summary = RAG_with_openai(prompt, model)
    else:
        print(f"Error: Unsupported provider '{provider}' specified.")
        return "// Unsupported provider."
    
    return summary

def main():
    parser = argparse.ArgumentParser(description="Add to or query the vector database.")
    parser.add_argument("mode", choices=["add", "query"], help="Mode to run: 'add' or 'query'")
    parser.add_argument("--pdf", type=str, help="Path to the PDF file for 'add' mode")
    parser.add_argument("--db_path", type=str, default="vector_db.index", help="Path to the vector DB file")
    parser.add_argument("--query", type=str, help="Search query for 'query' mode")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top results to retrieve")
    parser.add_argument("--model", type=str, default="openai", help="Model to use for response generation: 'groq', 'ollama', or 'openai'")
    parser.add_argument("--use_gpu", action='store_true', help="Use GPU for Faiss indexing and querying")


    args = parser.parse_args()

    if args.mode == "add":
        if not args.pdf:
            print("Error: PDF path is required in 'add' mode.")
            return
        add_pdf_to_vector_db(args.pdf, db_path=args.db_path, use_llama=True)
    elif args.mode == "query":
        if not args.query:
            print("Error: Query is required in 'query' mode.")
            return
        query_vector_db(args.db_path, args.query, top_k=args.top_k, model=args.model, use_gpu=args.use_gpu)

if __name__ == "__main__":
    main()
