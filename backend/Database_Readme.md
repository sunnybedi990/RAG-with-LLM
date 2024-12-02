### **Vector Databases Overview**

This document provides detailed instructions and configurations for the vector databases supported in the RAG system. These databases allow efficient storage and retrieval of vector embeddings for querying and summarization tasks.

---

## **Supported Vector Databases**

1. **FAISS**: Lightweight and efficient for local setups and GPU acceleration.
2. **Milvus**: Scalable and distributed, suitable for large-scale applications.
3. **Pinecone**: Cloud-native, fully managed for production environments.
4. **Qdrant**: Flexible with support for local, cloud, and in-memory setups.
5. **Weaviate**: Customizable with semantic search features.

---

## **Database Configuration**

Each vector database requires specific configurations. These can be set in a `config.yaml` file under the `vector_databases` section.

### **1. FAISS**
#### Description:
- Local database optimized for CPU/GPU.
- Best for lightweight applications.

#### Configuration:
```yaml
vector_databases:
  faiss:
    use_gpu: true  # Set to false for CPU-only
```

#### Key Features:
- No external service required.
- High performance for similarity search.

---

### **2. Milvus**
#### Description:
- Distributed and scalable vector database.
- Ideal for applications requiring cluster support.

#### Configuration:
```yaml
vector_databases:
  milvus:
    host: "localhost"
    port: 19530
    collection_name: "auto_generated"
    dimension: "auto_generated"
```

#### Setup Instructions:
1. Install Milvus dependencies:
   ```bash
   docker-compose -f milvus-standalone-docker-compose.yml up
   ```
2. Ensure `etcd` and `minio` are configured for distributed deployments.

---

### **3. Pinecone**
#### Description:
- Cloud-native, fully managed vector database.
- Simplifies deployments for production use.

#### Configuration:
```yaml
vector_databases:
  pinecone:
    api_key: "your_api_key"
    environment: "us-west-1"
    index_name: "auto_generated"
```

#### Setup Instructions:
1. Get an API key from Pinecone's dashboard.
2. Set the `environment` based on your account configuration.

---

### **4. Qdrant**
#### Description:
- Open-source, flexible vector database.
- Supports local, cloud, and in-memory deployments.

#### Configuration:
```yaml
vector_databases:
  qdrant:
    mode: "local"  # Options: "local", "cloud", "memory"
    host: "localhost"
    port: 6333
    collection_name: "auto_generated"
    dimension: "auto_generated"
```

#### Setup Instructions:
1. Start Qdrant in the chosen mode:
   - Local:
     ```bash
     docker run -p 6333:6333 qdrant/qdrant
     ```
   - In-Memory:
     No additional setup is required.

2. Ensure the collection is initialized only when missing.

---

### **5. Weaviate**
#### Description:
- Class-based semantic vector search.
- Integrates with various ML pipelines.

#### Configuration:
```yaml
vector_databases:
  weaviate:
    mode: "local"  # Options: "local", "cloud"
    host: "http://localhost:8080"
    class_name: "auto_generated"
    dimension: "auto_generated"
```

#### Setup Instructions:
1. For local mode:
   ```bash
   docker run -d -p 8080:8080 semitechnologies/weaviate
   ```
2. Configure cloud mode with the appropriate host and credentials.

---

## **Key Methods for All Databases**

### **Add Embeddings**
- **Function**: Adds vector embeddings to the database.
- **Parameters**:
  - `ids`: Unique IDs for the embeddings.
  - `embeddings`: List of vectors.
  - `metadata`: Optional metadata for each vector.

### **Search Embeddings**
- **Function**: Finds the top `k` nearest embeddings.
- **Parameters**:
  - `query_embedding`: Query vector for similarity search.
  - `top_k`: Number of nearest neighbors to return.

---

## **Example Usage**

### Adding Embeddings
```python
db.add_embeddings(
    ids=["id1", "id2"],
    embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    metadata=[{"key": "value1"}, {"key": "value2"}]
)
```

### Searching Embeddings
```python
results = db.search(
    query_embedding=[0.1, 0.2, 0.3],
    top_k=5
)
print(results)
```

---

## **Troubleshooting**

### FAISS
- Ensure the correct `faiss` library (`faiss-gpu` or `faiss-cpu`) is installed.

### Milvus
- Ensure dependent services (`etcd`, `minio`) are running for distributed mode.

### Pinecone
- Verify the API key and environment in `config.yaml`.

### Qdrant
- Check if the collection exists before recreating it.
- For memory mode, no external setup is required.

### Weaviate
- Ensure the Docker container is running for local mode.
- Verify class configurations in `config.yaml`.


