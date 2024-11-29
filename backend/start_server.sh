#!/bin/bash

# Default to Flask if no argument is provided
SERVER=${RAG_SERVER:-flask}

echo "Starting backend server with: $SERVER"

if [ "$SERVER" == "flask" ]; then
    echo "Running Flask backend (RAG.py)..."
    exec /opt/conda/envs/rag_env/bin/python backend/src/RAG.py
elif [ "$SERVER" == "fastapi" ]; then
    echo "Running FastAPI backend (RAG_fastapi.py)..."
    exec /opt/conda/envs/rag_env/bin/python backend/src/RAG_fastapi.py
else
    echo "Invalid server option: $SERVER. Use 'flask' or 'fastapi'."
    exit 1
fi
