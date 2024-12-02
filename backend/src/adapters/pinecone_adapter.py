import os
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import re

def sanitize_index_name(name):
    """
    Sanitizes the index name to conform to Pinecone's requirements.
    Converts to lowercase, replaces invalid characters with hyphens,
    and ensures the name doesn't start or end with a hyphen.
    """
    sanitized = re.sub(r'[^a-z0-9-]', '-', name.lower())
    sanitized = re.sub(r'^-+|-+$', '', sanitized)  # Remove leading/trailing hyphens
    return sanitized


class PineconeVectorDB:
    def __init__(self, api_key=None, environment="us-east-1", index_name="vector_index",dimension=768):
        if not api_key:
            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                raise ValueError("Pinecone API key not found. Set it in the environment or pass it explicitly.")
        index_name = sanitize_index_name(index_name)
        print(f"Initializing Pinecone index '{index_name}' with dimension: {dimension}")

        # Initialize Pinecone client
        self.pinecone = Pinecone(api_key=api_key)

        # Check if the index exists; create if it doesn't
        if index_name not in self.pinecone.list_indexes().names():
            print(f"Creating index '{index_name}'...")
            self.pinecone.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",  # Metric can be 'euclidean', 'cosine', or 'dotproduct'
                spec=ServerlessSpec(
                    cloud="aws",  # Specify the cloud provider
                    region=environment  # Specify the environment/region
                )
            )
        print(index_name)
        # Connect to the index
        self.index = self.pinecone.Index(index_name)

    def add_embeddings(self, embeddings, texts, namespace="default-namespace", metadata_key="text"):
        """
        Adds embeddings to the Pinecone index, including metadata.

        Args:
            embeddings (list): A list of embeddings (e.g., vectors).
            texts (list): A list of corresponding texts.
            namespace (str): Namespace for grouping vectors in Pinecone.
            metadata_key (str): Key under which text will be stored as metadata.
        """
        vectors = [
            {
                "id": f"vec-{i}",
                "values": embedding.tolist(),  # Convert numpy array to list
                "metadata": {metadata_key: text}  # Store the text as metadata
            }
            for i, (embedding, text) in enumerate(zip(embeddings, texts))
        ]

        # Use Pinecone's upsert API to insert the vectors
        upsert_response = self.index.upsert(
            vectors=vectors,
            namespace=namespace
        )

        print(f"Pinecone upsert response: {upsert_response}")



    def search(self, query_embedding, top_k=5,namespace="default-namespace"):
        """
        Searches for nearest neighbors in the Pinecone index.
        
        Args:
            query_embedding (list or np.ndarray): The query vector for the search.
            top_k (int): The number of nearest neighbors to retrieve.

        Returns:
            list of tuples: Each tuple contains (text or id, score) from the search results.
        """
        try:
            # Ensure the query embedding is a list for compatibility with Pinecone
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()

            # Perform the search
            response = self.index.query(namespace=namespace,vector=query_embedding, top_k=top_k, include_values=True, include_metadata=True)

            # Extract matches and process results
            matches = response.get('matches', [])
            results = [
                (
                    match['metadata'].get('text', match['id']),  # Default to ID if 'text' is unavailable
                    match['score']
                )
                for match in matches
            ]

            # Debugging: Print raw response and processed results
            print(f"Processed results: {results}")

            return results

        except Exception as e:
            raise RuntimeError(f"Error during Pinecone search: {e}")

    def get_all(self, namespace="default-namespace"):
        """
        Retrieve all vectors and metadata from the Pinecone index.
        """
        try:
            query_response = self.index.query(
                vector=[0] * self.dimension,  # Dummy vector to trigger retrieval
                top_k=10000,  # Adjust as needed based on index size
                include_metadata=True,
                namespace=namespace
            )
            return [item["metadata"]["text"] for item in query_response["matches"] if "text" in item["metadata"]]

        except Exception as e:
            raise RuntimeError(f"Error retrieving all records from Pinecone: {e}")

