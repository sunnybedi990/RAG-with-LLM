import weaviate
import os
import weaviate.classes as wvc

def connect_to_weaviate_cloud(cluster_url, api_key):
    """
    Connects to a Weaviate Cloud instance.
    """
    if not cluster_url or not api_key:
        raise ValueError("Both cluster_url and api_key must be provided for cloud connection.")
    
    try:
        client = weaviate.Client(
            url=cluster_url,
            auth_client_secret=wvc.AuthApiKey(api_key),
        )
        # Check connection
        if not client.is_ready():
            raise RuntimeError("Weaviate Cloud is not ready.")
        print("Connected to Weaviate Cloud successfully.")
        return client
    except Exception as e:
        raise RuntimeError(f"Error connecting to Weaviate Cloud: {e}")


class WeaviateVectorDB:
    def __init__(self, mode="local", host="http://localhost:8080", class_name="VectorObject", dimension=768):
        """
        Initializes a connection to a Weaviate instance (local or cloud).
        :param mode: "local" or "cloud" to specify connection type.
        :param host: The URL of the local Weaviate instance (ignored for cloud mode).
        :param class_name: The name of the class in the Weaviate schema.
        :param dimension: The dimension of vectors stored.
        :param cloud_config: Dictionary containing `cluster_url` and `api_key` for cloud mode.
        """
        self.class_name = class_name
        self.dimension = dimension

        # Connect to Weaviate based on mode
        if mode == "cloud":
            self.client = connect_to_weaviate_cloud(
                cluster_url=os.get("WEAVIATE_CLUSTER_URL"),
                api_key=os.get("WEAVIATE_API_KEY"),
            )
        elif mode == "local":
            self.client = weaviate.Client(url=host)
        else:
            raise ValueError("Mode must be either 'local' or 'cloud'.")

        self._initialize_schema()

    def _initialize_schema(self):
        schema = self.client.schema.get()
        if not any(cls["class"] == self.class_name for cls in schema.get("classes", [])):
            self.client.schema.create_class({
                "class": self.class_name,
                "vectorizer": "none",  # Using custom vectors
                "properties": [],
            })
        else:
            print(f"Class {self.class_name} already exists.")

    def add_embeddings(self, ids, embeddings):
        try:
            for id_, embedding in zip(ids, embeddings):
                if len(embedding) != self.dimension:
                    raise ValueError(f"Embedding dimension mismatch. Expected {self.dimension}, got {len(embedding)}")
                self.client.data_object.create(
                    data_object={"id": id_},
                    class_name=self.class_name,
                    vector=embedding,
                )
            print(f"Successfully added {len(ids)} embeddings to {self.class_name}.")
        except Exception as e:
            print(f"Error adding embeddings: {e}")

    def search(self, query_embedding, top_k=5):
        try:
            response = self.client.query.get(self.class_name, ["id"]).with_near_vector({"vector": query_embedding}).with_limit(top_k).do()
            matches = response.get("data", {}).get("Get", {}).get(self.class_name, [])
            results = [(match["id"], match["_additional"]["distance"]) for match in matches]
            return results
        except Exception as e:
            print(f"Error during search: {e}")
            return []

    def test_connection(self):
        try:
            if self.client.is_ready():
                print("Weaviate is ready.")
        except Exception as e:
            print(f"Connection failed: {e}")

    def get_all(self):
        """
        Retrieve all objects and vectors from the Weaviate class.
        """
        try:
            results = []
            response = self.client.query.get(self.class_name, ["id"]).with_limit(10000).with_additional(["vector"]).do()
            objects = response.get("data", {}).get("Get", {}).get(self.class_name, [])
            for obj in objects:
                 # Extract only the 'text' field if available
                if "text" in obj:
                    results.append(obj["text"])
            return results
        except Exception as e:
            raise RuntimeError(f"Error retrieving all records from Weaviate: {e}")
