# RAG System for Document Query and Summarization

This project is a Retrieval-Augmented Generation (RAG) system that allows users to upload documents, perform queries, and generate summaries using vector-based search and language models. The project consists of two main parts:
- **Backend**: Python-based server with vector database management and model integration.
- **Frontend**: React-based user interface for document upload, querying, and summarization.

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
7. [Troubleshooting](#troubleshooting)

---

## Project Overview
This RAG system uses vector-based search to index documents and allows users to perform complex queries or request summaries. The frontend provides a user-friendly interface for interactions, while the backend manages the vector database and calls the language model for generating responses.

## Setup and Installation

### Prerequisites
- **Conda** (for managing the backend environment with faiss-gpu)
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
   - Ensure `faiss-gpu` is installed via the Conda environment to leverage GPU support.

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
To create a Docker container for this RAG system:

1. **Build the Docker image**:
   ```bash
   docker build -t rag-system .
   ```

2. **Run the Docker container**:
   ```bash
   docker run -p 5000:5000 -p 3000:3000 rag-system
   ```
   - This command will expose both backend (`5000`) and frontend (`3000`) ports.
   - Ensure any necessary `.env` configuration is included in the Docker image, or use Docker secrets for sensitive information.

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
   - **Request**: Form-data with a file input named `pdf`.
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
   - **Description**: Summarizes the entire document based on the vector embeddings.
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
USE_GPU=true  # Set to true if using a GPU
VECTOR_DB_PATH=vector_dbs/  # Path to store vector databases
```

### Frontend `.env` File Example
In the frontend directory, create a `.env` file to specify the backend server URL:
```env
REACT_APP_BACKEND_URL=http://localhost:5000
```

---

## Troubleshooting
1. **Common Errors**:
   - Ensure that both backend and frontend are running on the expected ports (`5000` for backend and `3000` for frontend).
   - For CORS issues, confirm the backend allows requests from `http://localhost:3000`.

2. **Frontend Fails to Load or Shows Blank Page**:
   - Ensure that the backend is running and accessible at `http://localhost:5000`.

3. **Docker Configuration**:
   - Check if all necessary files and environment configurations are included in the Docker image.
   - Use `docker logs <container_id>` to troubleshoot any runtime issues in Docker.

For any other issues, please refer to the respective `README.md` files in each folder.