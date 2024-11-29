# Use a standard Ubuntu base image for CPU
FROM ubuntu:22.04

ENV USE_GPU=False
ENV PATH /opt/conda/bin:$PATH
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y wget curl bzip2 libglib2.0-0 libgl1-mesa-glx gnupg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Miniconda and set up Python environment with CPU-specific packages
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -a

COPY backend/requirements.txt .
RUN /opt/conda/bin/conda create -n rag_env python=3.11 -y && \
    /opt/conda/bin/conda run -n rag_env pip install --upgrade pip && \
    /opt/conda/bin/conda run -n rag_env pip install -r requirements.txt && \
    /opt/conda/bin/conda run -n rag_env conda install -c pytorch faiss-cpu=1.9.0 -y && \
    /opt/conda/bin/conda clean --all

COPY . .
EXPOSE 5000
# Add the script for choosing between Flask and FastAPI
COPY backend/start_server.sh /app/start_server.sh
RUN chmod +x /app/start_server.sh

# Default to Flask; can override with environment variable RAG_SERVER
CMD ["/app/start_server.sh"]
