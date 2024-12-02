import os
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

class MilvusVectorDB:
    def __init__(self, collection_name, dimension=768, host="localhost", port="19530"):
        """
        Initialize the Milvus vector database with a unique collection name based on the file name.
        """
        # Use the file name (without extension) as the collection name
        self.collection_name = collection_name
        self.dimension = dimension
        connections.connect(host=host, port=port)
        self.collection = self._initialize_or_load_collection()

    def _initialize_or_load_collection(self):
        """
        Initialize or load the collection in Milvus.
        """
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
        ]
        schema = CollectionSchema(fields)
        return Collection(name=self.collection_name, schema=schema)

    def add_embeddings(self, ids, embeddings):
        """
        Add embeddings and IDs to the collection.
        """
        self.collection.insert([ids, embeddings])

    def search(self, query_embedding, top_k=5):
        """
        Search for nearest neighbors to the given query embedding.
        """
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=top_k,
        )
        return results

    def drop_collection(self):
        """
        Drop the current collection.
        """
        if self.collection:
            print(f"Dropping collection: {self.collection_name}")
            self.collection.drop()

    def flush_collection(self):
        """
        Flush the current collection to persist data to storage.
        """
        self.collection.flush()
        print(f"Collection {self.collection_name} has been flushed to disk.")

    def get_all(self):
        """
        Retrieve all embeddings and associated metadata from the Milvus collection.
        """
        try:
            results = self.collection.query(expr="*", output_fields=["id", "embedding"])
            return [item["text"] for item in results]
        except Exception as e:
            raise RuntimeError(f"Error retrieving all records from Milvus: {e}")
