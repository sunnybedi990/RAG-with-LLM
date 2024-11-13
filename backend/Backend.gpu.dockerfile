# # Start from NVIDIA's official PyTorch container for CUDA compatibility
# FROM nvcr.io/nvidia/pytorch:23.04-py3

# # Set environment variables for GPU usage and Python environment management
# ENV USE_GPU=True
# ENV PATH /opt/conda/bin:$PATH

# # Set the working directory
# WORKDIR /app

# # Install essential libraries, including curl and gpg for NVIDIA setup if GPU support is enabled
# RUN apt-get update && \
#     apt-get install -y libglib2.0-0 libgl1-mesa-glx wget bzip2 gnupg curl && \
#     if [ "$USE_GPU" = "True" ]; then \
#         curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
#         |  gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && \
#         curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
#         | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
#         |  tee /etc/apt/sources.list.d/nvidia-container-toolkit.list && \
#         apt-get update && \
#         apt-get install -y nvidia-container-toolkit || echo "NVIDIA toolkit installation failed"; \
#     fi
# # Install Miniconda for Python environment management
# RUN apt-get update && \
#     apt-get install -y wget bzip2 && \
#     wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
#     bash ~/miniconda.sh -b -p /opt/conda && \
#     rm ~/miniconda.sh && \
#     apt-get clean && rm -rf /var/lib/apt/lists/*

# # Create a Conda environment and install dependencies
# COPY requirements.txt .
# RUN /opt/conda/bin/conda create -n rag_env python=3.11 -y && \
#     /opt/conda/bin/conda run -n rag_env pip install --upgrade pip && \
#     /opt/conda/bin/conda run -n rag_env pip install -r requirements.txt && \
#     if [ "$USE_GPU" = "True" ]; then \
#       /opt/conda/bin/conda run -n rag_env conda install -c pytorch faiss-gpu=1.9.0 -y; \
#     else \
#       /opt/conda/bin/conda run -n rag_env conda install -c pytorch faiss-cpu=1.9.0 -y; \
#     fi && \
#     /opt/conda/bin/conda clean --all

# # Copy the application code into the container
# COPY . .

# # Expose backend port for API access
# EXPOSE 5000

# # Activate the environment and run the backend server
# CMD ["/opt/conda/envs/rag_env/bin/python", "src/RAG.py"]

# Start from CUDA base image with Ubuntu and Miniconda for GPU compatibility
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set environment variables for GPU usage and Python environment management
ENV USE_GPU=True
ENV PATH /opt/conda/bin:$PATH

# Set the working directory
WORKDIR /app

# Install essential libraries and NVIDIA Container Toolkit for GPU support if enabled
RUN apt-get update && \
    apt-get install -y wget curl bzip2 libglib2.0-0 libgl1-mesa-glx gnupg && \
    if [ "$USE_GPU" = "True" ]; then \
        # Install NVIDIA Container Toolkit
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && \
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/ubuntu22.04/$(ARCH)/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list && \
        apt-get update && \
        apt-get install -y nvidia-container-toolkit && \
        apt-get clean && rm -rf /var/lib/apt/lists/*; \
    fi

# Install Miniconda for Python environment management
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -a

# Copy the requirements file into the container
COPY requirements.txt .

# Create a Conda environment, install dependencies, and handle GPU/CPU-specific packages
RUN /opt/conda/bin/conda create -n rag_env python=3.11 -y && \
    /opt/conda/bin/conda run -n rag_env pip install --upgrade pip && \
    /opt/conda/bin/conda run -n rag_env pip install -r requirements.txt && \
    if [ "$USE_GPU" = "True" ]; then \
      /opt/conda/bin/conda run -n rag_env conda install -c pytorch faiss-gpu=1.9.0 -y; \
    else \
      /opt/conda/bin/conda run -n rag_env conda install -c pytorch faiss-cpu=1.9.0 -y; \
    fi && \
    /opt/conda/bin/conda clean --all

# Copy the application code into the container
COPY . .

# Expose backend port for API access
EXPOSE 5000

# Run the backend server with the Conda environment
CMD ["/opt/conda/envs/rag_env/bin/python", "src/RAG.py"]
