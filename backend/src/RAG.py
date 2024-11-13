import os
import subprocess
import time
import traceback
from flask import Flask, jsonify, request, Response, stream_with_context
from flask_cors import CORS
from main import query_vector_db, add_pdf_to_vector_db, summarize_with_llm  # Ensure these functions are imported
from ollama import Client
import docker
import json

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
USE_GPU = os.getenv("USE_GPU", False).lower() == "true"
print(USE_GPU)
# Initialize Ollama Client
ollama_client = Client(host='http://ollama:11434')
# Define model mapping
def map_model_name(model_name):
    model_mapping = {
        "Llama 3.1 - 8B": "llama3.1:8b",
        "Llama 3.1 - 70B": "llama3.1:70b",
        "Gemma 2 - 2B": "gemma2:2b",
        "Gemma 2 - 9B": "gemma2:9b",
        "Mistral-Nemo - 12B": "mistral-nemo:12b",
        "Mistral Large 2 - 123B": "mistral-large2:123b",
        "Qwen 2 - 0.5B": "qwen2:0.5b",
        "Qwen 2 - 72B": "qwen2:72b",
        "DeepSeek-Coder V2 - 16B": "deepseek-coder-v2:16b",
        "Phi-3 - 3B": "phi3:3b",
        "Phi-3 - 14B": "phi3:14b",
    }
    return model_mapping.get(model_name, model_name)

# Track processes to manage cancellation
processes = {}
def is_model_installed(model_name):
    try:
        cli_model_name = map_model_name(model_name)
        models_response = ollama_client.list()  # List available models
        print(models_response)  # For debugging, this will show the structure of the response

        # Access the list of models from the dictionary
        models_list = models_response.get('models', [])
        
        for model in models_list:
            if isinstance(model, dict) and model.get('name') == cli_model_name:
                return True
        return False
    except Exception as e:
        print(f"Error checking model: {e}")
        print(traceback.format_exc())
        return False


def pull_model(model_name):
    cli_model_name = map_model_name(model_name)
    try:
        # Start pulling the model
        for line in ollama_client.pull(cli_model_name):
            yield line  # Stream each line to the client
            time.sleep(0.1)
        yield 'Model pull completed.'
    except Exception as e:
        yield f"Error pulling model: {e}"
        
# def pull_model(model_name):
#     cli_model_name = map_model_name(model_name)
#     client = docker.from_env()  # Initialize Docker client
#     print(cli_model_name)
#     try:
#         # Get the ollama container
#         #container = client.containers.get("ollama")
        
#         # Check if the container is running
#         # if container.status != "running":
#         #     print("Starting the ollama container...")
#         #     container.start()  # Start the container if it's not running
#         #     time.sleep(5)  # Allow some time for the container to initialize

#         # Run the pull command in the ollama container
#         #exec_result = container.exec_run(f"ollama pull {cli_model_name}", stream=True)

#         # Stream the output line-by-line
#         # for line in ollama_client.pull(cli_model_name):
#         #     yield line + "\n"
#         #     time.sleep(0.1)  # Optional delay for smoother streaming
#         ollama_client.pull(cli_model_name)
#         yield "Model pull completed.\n"

#     except Exception as e:
#         error_message = f"Error during model pull: {e}\n"
#         print(error_message)
#         yield error_message


def delete_model(model_name):
    try:
        cli_model_name = map_model_name(model_name)
        ollama_client.delete(cli_model_name)  # Delete the model
        return f"Model '{cli_model_name}' deleted."
    except Exception as e:
        return f"Error deleting model: {e}"
    
@app.route('/api/check-model', methods=['GET'])
def check_model():
    model = request.args.get('model')
    if not model:
        return jsonify({'error': 'Model name is required'}), 400

    if is_model_installed(model):
        return jsonify({'installed': True})
    else:
        return jsonify({'installed': False})

@app.route('/api/pull-model', methods=['POST'])
def pull_model_route():
    data = request.get_json()
    model = data.get('model')
    if not model:
        return jsonify({'error': 'Model name is required'}), 400

    # Directly return pull output to frontend
    return Response(stream_with_context(pull_model(model)), mimetype='text/plain')

@app.route('/api/delete-model', methods=['POST'])
def delete_model_route():
    data = request.get_json()
    model = data.get('model')
    if not model:
        return jsonify({'error': 'Model name is required'}), 400

    message = delete_model(model)
    return jsonify({'message': message})

@app.route('/api/cancel-pull', methods=['POST'])
def cancel_model_pull():
    data = request.get_json()
    model = data.get('model')
    cli_model_name = map_model_name(model)
    process = processes.get(cli_model_name)
    
    if process:
        process.terminate()
        processes.pop(cli_model_name, None)
        return jsonify({'success': True, 'message': f'Pull for {model} canceled.'})
    
    return jsonify({'success': False, 'message': f'No pull process found for {model}.'}), 404

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    provider = data.get('provider')
    embedding_provider = data.get('embedding_provider')
    embedding_model = data.get('embedding_model')
    query_text = data.get('query')
    model = data.get('model', 'openai')
    top_k = data.get('top_k', 3)

    db_path = f"/app/data/vector_db_{os.path.splitext(data.get('db_filename', 'default'))[0]}.index"

    try:
        cli_model_name = map_model_name(model) if provider == 'ollama' else model
        response_text = query_vector_db(
            db_path=db_path,
            query=query_text,
            top_k=top_k,
            model=cli_model_name,
            provider=provider,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            use_gpu=USE_GPU
        )
        return jsonify({"response": response_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/add', methods=['POST'])
def add():
    if 'pdf' not in request.files:
        return jsonify({"error": "PDF file is required"}), 400

    pdf_file = request.files['pdf']
    use_llama = request.form.get('use_llama', 'false').lower() == 'true'
    embedding_provider = request.form.get('embedding_provider')
    embedding_model = request.form.get('embedding_model')

    db_path = f"/app/data/vector_db_{os.path.splitext(pdf_file.filename)[0]}.index"
    pdf_path = os.path.join("/tmp", pdf_file.filename)
    pdf_file.save(pdf_path)
    print('USE GPU',USE_GPU)
    try:
        add_pdf_to_vector_db(
            pdf_path=pdf_path,
            db_path=db_path,
            use_llama=use_llama,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            use_gpu=USE_GPU
        )
        return jsonify({"message": f"Document added to vector database at {db_path}."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    provider = data.get('provider')
    model = data.get('model', 'openai')
    embedding_provider = data.get('embedding_provider')
    embedding_model = data.get('embedding_model')
    db_filename = data.get('db_filename')

    db_path = f"/app/data/vector_db_{os.path.splitext(db_filename)[0]}.index"

    try:
        cli_model_name = map_model_name(model) if provider == 'ollama' else model
        chunks = query_vector_db(
            db_path=db_path,
            query="*",
            top_k=20,
            model=cli_model_name,
            provider=provider,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            use_gpu=USE_GPU
        )
        summary_text = summarize_with_llm(chunks, model, provider)
        return jsonify({"summary": summary_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# @app.route('/query', methods=['POST'])
# def query():
#     data = request.json
#     provider = data.get('provider')
#     query_text = data.get('query')
#     model = data.get('model', 'openai')
#     top_k = data.get('top_k', 3)

#     # Use the uploaded file's name to determine db_path
#     db_path = f"/app/data/vector_db_{os.path.splitext(data.get('db_filename', 'default'))[0]}.index"

#     try:
#         if provider == 'ollama':
#             cli_model_name = map_model_name(model)
#         else:
#             cli_model_name = model
#         response_text = query_vector_db(db_path=db_path, query=query_text, top_k=top_k, model=cli_model_name,provider=provider,use_gpu=USE_GPU)
#         return jsonify({"response": response_text})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# # Add endpoint for adding a PDF document to the vector database
# @app.route('/add', methods=['POST'])
# def add():
#     if 'pdf' not in request.files:
#         return jsonify({"error": "PDF file is required"}), 400

#     pdf_file = request.files['pdf']
#     use_llama = request.form.get('use_llama', 'true').lower() == 'true'

#     # Use the uploaded file's name as db_path
#     db_path = f"/app/data/vector_db_{os.path.splitext(pdf_file.filename)[0]}.index"
#     pdf_path = os.path.join("/tmp", pdf_file.filename)
#     pdf_file.save(pdf_path)

#     try:
#         add_pdf_to_vector_db(pdf_path=pdf_path, db_path=db_path, use_llama=use_llama,use_gpu=USE_GPU)
#         return jsonify({"message": f"Document added to vector database at {db_path}."})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
    
# # Route for summarization
# @app.route('/summarize', methods=['POST'])
# def summarize():
#     data = request.json
#     provider = data.get('provider')
#     model = data.get('model', 'openai')
#     db_filename = data.get('db_filename')

#     # Ensure the vector database file path is correctly specified
#     db_path = f"/app/data/vector_db_{os.path.splitext(db_filename)[0]}.index"

#     try:
#         if provider == 'ollama':
#             cli_model_name = map_model_name(model)
#         else:
#             cli_model_name = model
#         # Retrieve chunks from the vector database
#         chunks = query_vector_db(db_path=db_path, query="*", top_k=20, model=cli_model_name,use_gpu=USE_GPU)  # Retrieve top 20 chunks

#         # Create a summary from these chunks
#         summary_text = summarize_with_llm(chunks, model, provider)

#         return jsonify({"summary": summary_text})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
