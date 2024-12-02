from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import os

class QdrantVectorDB:
    def __init__(self, collection_name="vector_collection", dimension=768, mode="local", host="localhost", port=6333, path=None, api_key=None):
        """
        Initialize Qdrant client and collection.
        Args:
            collection_name (str): Name of the Qdrant collection.
            dimension (int): Dimensionality of the vectors.
            mode (str): Deployment mode ("local", "cloud", "memory").
            host (str): Host of the Qdrant service (for "local").
            port (int): Port of the Qdrant service (for "local").
            path (str, optional): Path to the Qdrant database (for "local").
            api_key (str, optional): API key for Qdrant Cloud (for "cloud").
        """
        self.collection_name = collection_name
        self.dimension = dimension
        
        if mode == "local":
            self.client = QdrantClient(host=host, port=port)
        elif mode == "cloud":
            if not api_key:
                raise ValueError("API key is required for cloud mode.")
            cluster_url = os.getenv('QDRANT_CLUSTER_URL')
            self.client = QdrantClient(url=cluster_url,api_key=api_key)
        elif mode == "memory":
            self.client = QdrantClient(":memory:")
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        self._initialize_collection()

    def _initialize_collection(self):
            """
            Initialize the Qdrant collection. If the collection already exists, connect to it.
            """
            try:
                # Check if the collection exists
                existing_collections = [col.name for col in self.client.get_collections().collections]
                
                if self.collection_name not in existing_collections:
                    # Create the collection only if it doesn't exist
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
                    )
                    print(f"Collection '{self.collection_name}' created.")
                else:
                    print(f"Collection '{self.collection_name}' already exists. Connected to the collection.")
            except Exception as e:
                raise ValueError(f"Error initializing or connecting to collection: {e}")


    def add_embeddings(self, ids, embeddings, metadata=None):
        """
        Add embeddings to the Qdrant collection.
        Args:
            ids (list): List of unique IDs for the embeddings.
            embeddings (list): List of embeddings (vectors).
            metadata (list, optional): List of dictionaries containing metadata for each vector.
        """
        if ids is None:
            ids = list(range(len(embeddings)))
            
        if metadata is None:
            metadata = [{}] * len(ids)  # Default to empty metadata for each vector

        points = [
            PointStruct(id=ids[i], vector=embeddings[i], payload=metadata[i])
            for i in range(len(ids))
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_embedding, top_k=5):
        """
        Search for the nearest neighbors in the Qdrant collection.
        Args:
            query_embedding (list): The query vector for searching.
            top_k (int): Number of nearest neighbors to return.
        Returns:
            list: List of search results with ID, score, and payload.
        """
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True,
        )
        print(results)
        return [
                {"id": point.id, "score": point.score, "payload": point.payload}
                for point in results
            ]
        
    def get_all(self):
        """
        Retrieve all embeddings and metadata from the Qdrant collection.
        """
        try:
            results = []
            scroll_id = None
            while True:
                response = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=None,
                    limit=1000,
                    with_payload=True,
                    offset=scroll_id
                )
                # Extract only the 'text' field from payloads
                results.extend([
                    point.payload.get("text", "") for point in response.points if "text" in point.payload
                ])
                scroll_id = response.offset
                if not scroll_id:
                    break
            return results
        except Exception as e:
            raise RuntimeError(f"Error retrieving all records from Qdrant: {e}")
