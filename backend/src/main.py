import argparse
from VectorDB import VectorDB
from add_to_vector_db import add_pdf_to_vector_db
from llm_response.llm_utils import generate_response
from llm_response.chart_parser import parse_response_and_generate_chart
from llm_response.prompt import Prompt

def query_vector_db(db_path, db_type,db_config, query, top_k=5, model="openai", provider='', embedding_provider='', embedding_model='', use_gpu=False):
    """
    Performs a query on the vector database and generates a response using the specified LLM.
    """
    print(db_type)
    # Initialize the vector database
    vector_db = VectorDB(
        db_path=db_path,
        db_type=db_type,
        provider=embedding_provider,
        model_name=embedding_model,
        use_gpu=use_gpu,
        index_name=db_path,
        db_config=db_config
    )

    # Load the database index if supported
    if hasattr(vector_db.db, "load_index"):
        vector_db.load_index(db_path)

    print(f"Querying the vector database with: '{query}'")
    if query == "*":
        # Retrieve all documents
        results = vector_db.get_all()
        print('Result shape', results)
        # # Clean results for summarization
        if isinstance(results[0], tuple) and len(results[0]) == 2:
            # Remove scores if necessary
            clean_results = [text for text, score in results]
        else:
            clean_results = [text for text in results]
        accumulate_resuls = ''.join(clean_results)
        return accumulate_resuls

    # Perform the search using VectorDB
    results = vector_db.search(query, top_k=top_k)
    print(f"Raw search results: {results}")

    if not results:
        print("No results found in the vector database.")
        return "No relevant information found."
    # Check if the query asks for a graph or chart
    is_graph_request = "graph" in query.lower() or "chart" in query.lower()

    # Format the results into a prompt for the LLM
    if isinstance(results[0], tuple) and len(results[0]) == 2:
        # Results include both text and score
        formatted_results = "\n".join([f"{text} (score: {score:.2f})" for text, score in results])
    else:
        # Results include only text
        formatted_results = "\n".join([f"{text}" for text in results])
    
    task_type = "chart" if is_graph_request else "query"
    prompt = Prompt(query=query, context=formatted_results, task_type=task_type).generate_prompt()

    # Generate response using the specified LLM
    response = generate_response(prompt, model, provider)

    if is_graph_request:
        chart_path, chart_type = parse_response_and_generate_chart(response)
        if chart_path:
            return {"chart_type": chart_type, "chart_image_path": chart_path}
        else:
            return f"LLM response: {response}"
    # if is_graph_request:
    #     # Modify prompt for chart data
    #     prompt = f"User's query: {query}\nRelevant information:\n{formatted_results}\n" \
    #              "Based on the trend, suggest the best chart type and provide the data in JSON format suitable for creating the chart. Include the chart type and label."
    # else:
    #     # Standard prompt
    #     prompt = f"User's query: {query}\nRelevant information:\n{formatted_results}"

    # # Generate response using the specified LLM
    # response = generate_response(prompt, model, provider)

    # if is_graph_request:
    #     chart_path, chart_type = parse_response_and_generate_chart(response)
    #     if chart_path:
    #         return {"chart_type":chart_type,"chart_image_path": chart_path}
    #     else:
    #         return f"LLM response: {response}"
    print(response)
    return response


def summarize_with_llm(chunks, model, provider):
    """
    Summarizes document chunks using the specified LLM provider.
    Args:
        chunks (str): Combined text chunks for summarization.
        model (str): The LLM model to use.
        provider (str): The provider of the LLM (e.g., "ollama", "openai", "groq").
    Returns:
        str: Summary text generated by the LLM.
    """
    try:
        # Create a summarization prompt
        prompt = Prompt(context=chunks, task_type="summarization").generate_prompt()

        # Generate summary using the specified LLM provider
        return generate_response(prompt, model, provider)
    except Exception as e:
        print(f"Error during summarization: {e}")
        raise RuntimeError(f"Summarization failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Add to or query the vector database.")
    parser.add_argument("mode", choices=["add", "query"], help="Mode to run: 'add' or 'query'")
    parser.add_argument("--pdf", type=str, help="Path to the PDF file for 'add' mode")
    parser.add_argument("--db_path", type=str, default="vector_db.index", help="Path to the vector DB file")
    parser.add_argument("--db_type", type=str, default="faiss", choices=["faiss", "milvus", "pinecone", "qdrant", "weaviate"], help="Type of vector database")
    parser.add_argument("--query", type=str, help="Search query for 'query' mode")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top results to retrieve")
    parser.add_argument("--model", type=str, default="openai", help="Model to use for response generation: 'groq', 'ollama', or 'openai'")
    parser.add_argument("--use_gpu", action='store_true', help="Use GPU for Faiss indexing and querying")

    args = parser.parse_args()

    if args.mode == "add":
        if not args.pdf:
            print("Error: PDF path is required in 'add' mode.")
            return
        add_pdf_to_vector_db(
            args.pdf,
            db_path=args.db_path,
            db_type=args.db_type,
            embedding_provider=args.embedding_provider,
            embedding_model=args.embedding_model,
            use_gpu=args.use_gpu
        )
    elif args.mode == "query":
        if not args.query:
            print("Error: Query is required in 'query' mode.")
            return
        query_vector_db(
            args.db_path,
            db_type=args.db_type,
            db_config=None,  # Provide appropriate config if necessary
            query=args.query,
            top_k=args.top_k,
            model=args.model,
            provider=args.model,
            embedding_provider=args.embedding_provider,
            embedding_model=args.embedding_model,
            use_gpu=args.use_gpu
        )

if __name__ == "__main__":
    main()
