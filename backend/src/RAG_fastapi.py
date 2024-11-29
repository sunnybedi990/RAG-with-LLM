import os
import time
import traceback
from fastapi import FastAPI, HTTPException, UploadFile, Form, Depends, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from main import query_vector_db, add_pdf_to_vector_db, summarize_with_llm  # Ensure these functions are imported
from ollama import Client
import nest_asyncio

nest_asyncio.apply()

# Set the base directory to the backend folder
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Define the vector_dbs directory path
VECTOR_DBS_DIR = os.path.join(BASE_DIR, "vector_dbs")

# Ensure the directory exists
os.makedirs(VECTOR_DBS_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI()

# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

USE_GPU = os.getenv("USE_GPU", False).lower() == "true"
print(USE_GPU)

# Initialize Ollama Client
ollama_client = Client(host='http://ollama:11434')

# Define model mapping
def map_model_name(model_name: str) -> str:
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

# Utility functions
def is_model_installed(model_name: str) -> bool:
    try:
        cli_model_name = map_model_name(model_name)
        models_response = ollama_client.list()
        models_list = models_response.get('models', [])
        return any(model.get('name') == cli_model_name for model in models_list)
    except Exception as e:
        print(f"Error checking model: {e}")
        print(traceback.format_exc())
        return False


async def pull_model(model_name: str):
    cli_model_name = map_model_name(model_name)
    try:
        for line in ollama_client.pull(cli_model_name):
            yield line + "\n"
            time.sleep(0.1)
    except Exception as e:
        yield f"Error pulling model: {e}\n"

def delete_model(model_name: str) -> str:
    try:
        cli_model_name = map_model_name(model_name)
        ollama_client.delete(cli_model_name)
        return f"Model '{cli_model_name}' deleted."
    except Exception as e:
        return f"Error deleting model: {e}"

# Request and response models
class QueryRequest(BaseModel):
    provider: str
    embedding_provider: str
    embedding_model: str
    query: str
    model: str = "openai"
    top_k: int = 3
    db_filename: str

class SummarizeRequest(BaseModel):
    provider: str
    model: str = "openai"
    embedding_provider: str
    embedding_model: str
    db_filename: str

# Routes
@app.get("/api/check-model")
async def check_model(model: str):
    if is_model_installed(model):
        return {"installed": True}
    return {"installed": False}

@app.post("/api/pull-model")
async def pull_model_route(model: str):
    return StreamingResponse(pull_model(model), media_type="text/plain")

@app.post("/api/delete-model")
async def delete_model_route(model: str):
    message = delete_model(model)
    return {"message": message}

@app.post("/query")
async def query(data: QueryRequest):
    try:
        # Log incoming data
        print("Received request:", data.dict())

        cli_model_name = map_model_name(data.model) if data.provider == "ollama" else data.model
        db_path = os.path.join(VECTOR_DBS_DIR, f"vector_db_{os.path.splitext(data.db_filename.replace(' ', '_'))[0]}.index")
        print(db_path)
        # Perform query
        response_text = query_vector_db(
            db_path=db_path,
            query=data.query,
            top_k=data.top_k,
            model=cli_model_name,
            provider=data.provider,
            embedding_provider=data.embedding_provider,
            embedding_model=data.embedding_model,
            use_gpu=USE_GPU,
        )
        return {"response": response_text}
    except Exception as e:
        # Log the error for debugging
        print(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/add")
async def add(
    pdf: UploadFile = File(...),
    embedding_provider: str = Form(...),
    embedding_model: str = Form(...),
    parser_type: str = Form(...)
):
    try:
        pdf_path = f"/tmp/{pdf.filename}"
        with open(pdf_path, "wb") as f:
            f.write(await pdf.read())
        
        db_path = os.path.join(VECTOR_DBS_DIR, f"vector_db_{os.path.splitext(pdf.filename.replace(' ', '_'))[0]}.index")
        


        print(db_path)
        use_llama = parser_type == "LlamaParser"

        add_pdf_to_vector_db(
            pdf_path=pdf_path,
            db_path=db_path,
            use_llama=use_llama,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            use_gpu=USE_GPU,
        )
        print(f"FAISS index created at: {db_path}")

        if not os.path.exists(db_path):
            raise HTTPException(status_code=500, detail="Failed to create vector database.")
        return {"message": f"Document added to vector database at {db_path}."}
    except Exception as e:
        print(f"Error adding document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize")
async def summarize(data: SummarizeRequest):
    try:
        cli_model_name = map_model_name(data.model) if data.provider == "ollama" else data.model
        db_path = f"/app/data/vector_db_{os.path.splitext(data.db_filename)[0]}.index"
        
        chunks = query_vector_db(
            db_path=db_path,
            query="*",
            top_k=20,
            model=cli_model_name,
            provider=data.provider,
            embedding_provider=data.embedding_provider,
            embedding_model=data.embedding_model,
            use_gpu=USE_GPU,
        )
        summary_text = summarize_with_llm(chunks, data.model, data.provider)
        return {"summary": summary_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run FastAPI app with a server like Uvicorn for production
