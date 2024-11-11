# Start from Miniconda with Python 3.11 as the base for cross-platform compatibility
FROM continuumio/miniconda3

# Arguments to determine if the build should include GPU support
ARG USE_GPU=True

# Set working directory
WORKDIR /app

# Install essential libraries, including curl and gpg for NVIDIA setup if GPU support is enabled
RUN apt-get update && \
    apt-get install -y libglib2.0-0 libgl1-mesa-glx wget bzip2 gnupg curl && \
    if [ "$USE_GPU" = "True" ]; then \
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        |  gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && \
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        |  tee /etc/apt/sources.list.d/nvidia-container-toolkit.list && \
        apt-get update && \
        apt-get install -y nvidia-container-toolkit || echo "NVIDIA toolkit installation failed"; \
    fi

# Install faiss-gpu if GPU support is enabled, otherwise install faiss-cpu
RUN if [ "$USE_GPU" = "True" ]; then \
        conda install -c pytorch faiss-gpu=1.9.0 -y; \
    else \
        conda install -c pytorch faiss-cpu=1.9.0 -y; \
    fi && \
    conda clean --all

# Set environment variable for Python path
ENV PATH /opt/conda/bin:$PATH

# Create data directory with appropriate permissions
RUN mkdir -p /app/data && chmod -R 777 /app/data

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the backend code into the container
COPY . .

# Expose backend port
EXPOSE 5000

# Run backend server
CMD ["python", "src/RAG.py"]
