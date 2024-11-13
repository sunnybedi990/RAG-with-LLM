import sys
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import openai
from embedding_initializer import initialize_embedding_model


class VectorDB:
    def __init__(self, provider, model_name, use_gpu=True, api_key=None):
        """
        Initializes a FAISS index with the specified embedding provider and model.
        
        :param provider: Embedding provider name (e.g., 'sentence_transformers' or 'openai')
        :param model_name: Embedding model name (e.g., 'all-mpnet-base-v2')
        :param use_gpu: Whether to use GPU for FAISS
        :param api_key: Optional API key for OpenAI models
        """
        # Load configuration for the selected model
        self.model, self.dimension, self.iscallable = initialize_embedding_model(provider, model_name, api_key)
        self.use_gpu = use_gpu
        self.provider = provider
        self.model_name = model_name
        self.id_map = {}
        
        
        # Initialize FAISS index
        if self.use_gpu:
            self.res = faiss.StandardGpuResources()
            self.index = faiss.GpuIndexFlatL2(self.res, self.dimension)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)

    def verify_index(self):
        """Verifies that all embeddings were stored in the FAISS index."""
        print(f"Total entries in index: {self.index.ntotal}")
        print(f"Total entries in id_map: {len(self.id_map)}")
        if self.index.ntotal != len(self.id_map):
            print("Warning: Mismatch between index entries and id_map entries.")
        
        # Display a sample of stored entries
        if self.index.ntotal > 0:
            dummy_query = np.zeros((1, self.dimension), dtype='float32')
            distances, indices = self.index.search(dummy_query, min(5, self.index.ntotal))
            print("Sample stored entries in the index:")
            for i, idx in enumerate(indices[0]):
                if idx in self.id_map:
                    print(f"Index {idx}: {self.id_map[idx]} - Distance: {distances[0][i]}")
                else:
                    print(f"Index {idx} not found in id_map.")
        else:
            print("The index is empty.")

    def add_embeddings(self, texts, batch_size=32):
        """
        Adds embeddings for a list of texts to the FAISS index.
        
        :param texts: A list of text strings to embed and add to the index
        :param batch_size: Number of texts to process at once
        """
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            print(f"Processing batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")
            
            # Debug: Print batch texts
            print(f"Batch texts: {batch_texts}")

           # Generate embeddings with model
            embeddings = []
            for text in batch_texts:
                try:
                    print(f"Processing text: {text} (type: {type(text)})")
                    
                    # Generate embedding
                      # Use the appropriate method based on `is_callable` flag
                    if self.iscallable:
                        result = self.model(text)
                    else:
                        result = self.model.encode(text)
                    # Check if result is a dictionary
                    if isinstance(result, dict):
                        # Try common keys that may store embeddings
                        embedding = result.get("embedding") or result.get("data")
                        if isinstance(embedding, list):  # Handle OpenAI's response structure
                            embedding = embedding[0]['embedding'] if isinstance(embedding[0], dict) else embedding
                    else:
                        embedding = result  # Directly use if it's already a list or array
                    
                    # Convert embedding to numpy array
                    embedding_array = np.array(embedding, dtype='float32')
                    embeddings.append(embedding_array)
                    print(f"Generated embedding for text: {embedding_array.shape}")
                    
                except Exception as e:
                    print(f"Error generating embedding for text '{text}': {e}")
                    continue
            # Debug: Print generated embeddings
            print(f"Generated embeddings for batch {i // batch_size + 1}")
            print(type(embeddings))

            # Add to FAISS index
            try:
                embeddings_np = np.vstack(embeddings)
                self.index.add(embeddings_np)
            except Exception as e:
                print(f"Error adding embeddings to FAISS index: {e}")
                continue
            
            # Update id_map with correct indexing
            current_id = self.index.ntotal - len(embeddings)
            for text in batch_texts:
                self.id_map[current_id] = text
                current_id += 1
        
        print("All extracted data has been added to VectorDB.")
        print(f"Total entries in index: {self.index.ntotal}, Total in id_map: {len(self.id_map)}")


    def search(self, query, top_k=5):
        """
        Searches the FAISS index for the closest embeddings to the query.
        
        :param query: The query text to search for
        :param top_k: Number of top matches to return
        :return: List of top_k closest results
        """
        try:
            print(f"Processing query: {query} (type: {type(query)})")
            
            # Generate query embedding based on whether model is callable
            if self.iscallable:
                result = self.model(query)
            else:
                result = self.model.encode(query)
            
            # Check if result is a dictionary, similar to add_embeddings
            if isinstance(result, dict):
                query_embedding = result.get("embedding") or result.get("data")
                if isinstance(query_embedding, list):  # Handle OpenAI's response structure
                    query_embedding = query_embedding[0]['embedding'] if isinstance(query_embedding[0], dict) else query_embedding
            else:
                query_embedding = result  # Directly use if it's already a list or array
            
            # Convert to numpy array and reshape if necessary
            query_embedding = np.array(query_embedding, dtype='float32')
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Perform search in FAISS index
            distances, indices = self.index.search(query_embedding, top_k)
            
            # Retrieve results from id_map
            results = [(self.id_map[idx], dist) for idx, dist in zip(indices[0], distances[0]) if idx in self.id_map]
            
            print(f"Search results: {results}")
            return results
        
        except Exception as e:
            print(f"Error generating or searching embeddings for query '{query}': {e}")
            return []


    def save_index(self, path="faiss_index.index", id_map_path="id_map.pkl"):
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, path)
        else:
            faiss.write_index(self.index, path)
        with open(id_map_path, 'wb') as f:
            pickle.dump(self.id_map, f)
        print(f"Index saved to {path} and id_map saved to {id_map_path}.")

    def load_index(self, path="faiss_index.index", id_map_path="id_map.pkl"):
        cpu_index = faiss.read_index(path)
        self.index = faiss.index_cpu_to_gpu(self.res, 0, cpu_index) if self.use_gpu else cpu_index
        with open(id_map_path, 'rb') as f:
            self.id_map = pickle.load(f)
        print(f"Index loaded from {path} and id_map loaded from {id_map_path}.")
