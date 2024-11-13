# RAG System for Document Query and Summarization

This project is a Retrieval-Augmented Generation (RAG) system that enables users to upload documents, perform complex queries, and generate summaries using vector-based search and various language models. It consists of two main parts:
- **Backend**: A Python server for vector database management and model integration.
- **Frontend**: A React user interface for document upload, querying, and summarization.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Setup and Installation](#setup-and-installation)
   - [Backend Setup (Conda Environment)](#backend-setup-conda-environment)
   - [Frontend Setup](#frontend-setup)
   - [Docker Build](#docker-build)
3. [Usage](#usage)
   - [Starting the Backend](#starting-the-backend)
   - [Starting the Frontend](#starting-the-frontend)
4. [Project Structure](#project-structure)
5. [API Documentation](#api-documentation)
6. [Configuration](#configuration)
7. [Key Features and Sidebar Options](#key-features-and-sidebar-options)
8. [Troubleshooting](#troubleshooting)

---

## Project Overview
This RAG system uses vector-based search to index documents, allowing users to perform complex queries or request summaries. The frontend provides an interactive interface, while the backend manages the vector database and communicates with language models to generate responses.

## Setup and Installation

### Prerequisites
- **Conda** (for managing the backend environment)
- **Docker** (for containerized builds)
- **Node.js and npm** (for the frontend)
- **Git** (to clone the repository)

---

### Backend Setup (Conda Environment)
1. **Navigate to the backend directory**:
   ```bash
   cd project-root/backend
   ```

2. **Create and activate a Conda environment**:
   ```bash
   conda create -n rag-system python=3.8
   conda activate rag-system
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   - Ensure `faiss-gpu` is installed via Conda if using GPU support.

4. **Run the backend server**:
   ```bash
   python src/main.py
   ```
   The backend server will start on `http://localhost:5000`.

### Frontend Setup
1. **Navigate to the frontend directory**:
   ```bash
   cd project-root/frontend/rag-ui
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Run the frontend server**:
   ```bash
   npm start
   ```
   The frontend server will start on `http://localhost:3000`.


### Docker Build
To build and run a Docker container for this RAG system, set `USE_GPU` as a build argument to dynamically choose between GPU and CPU support:

1. **Build the Docker image**:
   - **For GPU** (`USE_GPU=true`):
     ```bash
     docker build --build-arg USE_GPU=true -f backend/Backend.gpu.dockerfile -t rag-system-gpu .
     ```
   - **For CPU** (`USE_GPU=false`):
     ```bash
     docker build --build-arg USE_GPU=false -f backend/Backend.cpu.dockerfile -t rag-system-cpu .
     ```

2. **Run the Docker container**:
   - **GPU setup**:
     ```bash
     docker-compose -f compose.gpu.yml up 
     ```
   - **CPU setup**:
     ```bash
     docker-compose -f compose.cpu.yml up
     ```

   - These commands will expose the backend (`5000`) and frontend (`3000`) ports.

**Note:** Ensure that `USE_GPU` is handled in the Dockerfile with conditional steps based on the value of `USE_GPU`, allowing the setup to dynamically install GPU or CPU dependencies as needed.

---

## Usage

### Starting the Backend
After setting up the backend, start the backend server:
```bash
python backend/src/main.py
```

### Starting the Frontend
Once the frontend setup is complete, start the frontend:
```bash
cd frontend/rag-ui
npm start
```

The application will be available at `http://localhost:3000`. You can upload documents, perform queries, and request summaries from the UI.

---

## Project Structure
```
project-root/
│
├── backend/                   # Backend (Python) directory
│   ├── vector_dbs/            # Vector DB index files
│   └── src/                   # Source code
│       ├── add_to_vector_db.py
│       ├── main.py
│       ├── RAG.py
│       ├── rag_models.py
│       └── VectorDB.py
│
├── frontend/                  # Frontend (React) directory
│   └── rag-ui/                # Main React app
│       ├── public/
│       ├── src/
│       ├── package.json
│       ├── .env
│
└── requirements.txt           # Backend dependencies
└── Dockerfile                 # Docker build file
```

---

## API Documentation

### Backend API Endpoints

#### 1. **Upload Document**
   - **Endpoint**: `/add`
   - **Method**: `POST`
   - **Description**: Uploads a PDF document, processes it, and adds it to the vector database.
   - **Request**: Form-data with a file input named `pdf`, as well as `embedding_provider`, `embedding_model`, and `parser_type`.
   - **Response**: Success or error message.

#### 2. **Query Document**
   - **Endpoint**: `/query`
   - **Method**: `POST`
   - **Description**: Queries the vector database and retrieves relevant information.
   - **Request JSON**:
     ```json
     {
       "provider": "ollama",
       "query": "Your query here",
       "model": "Llama 3.1 - 8B",
       "top_k": 3,
       "db_filename": "your_document.pdf"
     }
     ```
   - **Response**: Query result or error message.

#### 3. **Summarize Document**
   - **Endpoint**: `/summarize`
   - **Method**: `POST`
   - **Description**: Summarizes the entire document based on vector embeddings.
   - **Request JSON**:
     ```json
     {
       "provider": "ollama",
       "model": "Llama 3.1 - 8B",
       "db_filename": "your_document.pdf"
     }
     ```
   - **Response**: Summary result or error message.

---

## Configuration

### Backend `.env` File Example
Create a `.env` file in the backend directory to configure settings:

```env
VECTOR_DB_PATH=vector_dbs/  # Path to store vector databases
LLAMA_CLOUD_API_KEY=your_llama_api_key_here  # API key for Llama Cloud
OPENAI_API_KEY=your_openai_api_key_here  # API key for OpenAI
GROQ_API_KEY=your_groq_api_key_here  # API key for Groq
```

Make sure to replace `your_llama_api_key_here`, `your_openai_api_key_here`, and `your_groq_api_key_here` with your actual API keys. These keys are required to authenticate requests to their respective services for embedding and language model tasks.

### Frontend `.env` File Example
In the frontend directory, create a `.env` file to specify the backend server URL:
```env
REACT_APP_BACKEND_URL=http://localhost:5000
```

---

## Key Features and Sidebar Options

### Model Selection
- **API Model Provider**: Choose from multiple providers such as `openai`, `groq`, and `ollama`.
- **LLM Models**: Options like `gpt-4o`, `llama3-70b-8192`, and others.

### Embedding Models
Embedding models are organized into nested categories:
1. **Classic Models (pre-2020)**: Includes models like `elmo-original`, `bert-base-uncased`, `xlnet-base-cased`, etc.
2. **Sentence Transformer Models (2020–2022)**: Includes `all-mpnet-base-v2`, `all-MiniLM-L6-v2`, etc.
3. **LLM-Based Embeddings**: Contains models like `text-embedding-ada-002`, `gpt2-medium`, `embed-v3`, etc.
4. **Newer Specialized Models (2023–2024)**: Includes models like `arctic-embed-large`, `nv-embed`, `longembed`, etc.
5. **Mamba-Based State Space Models**: Options like `mamba-byte`, `moe-mamba`, and `vision-mamba`.
6. **Hugging Face General Embedding Models**: Options include `distilbert-base-uncased`, `roberta-base`.


### Parser Options
Users can choose between:
- **LlamaParser**: For complex documents.
- **Custom Parser (Fitz + Camelot)**: For simpler documents.

### Top K Results
Customize the number of top results returned for queries.

### Model Management
- **Download and Delete Models**: Check, download, or delete models as needed.
- **Cancel Downloads**: An option to cancel model downloads mid-way.

---

## Troubleshooting
1. **Common Errors**:
   - Confirm backend and frontend are running on expected ports (`5000` for backend, `3000` for frontend).
   - For CORS issues, ensure the backend allows requests from `http://localhost:3000`.

2. **Frontend Fails to Load or Shows Blank Page**:
   - Ensure the backend is running and accessible at `http://localhost:5000`.

3. **Docker Configuration**:
   - Check `.env` files and necessary configurations are in the Docker image.
   - Use `docker logs <container_id>` for debugging runtime issues in Docker.

For any other issues, refer to the `README.md` files in each folder.