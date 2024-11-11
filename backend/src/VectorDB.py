import sys
sys.setrecursionlimit(2000)
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle

class VectorDB:
    def __init__(self, dimension=384,use_gpu=True):
        
        """
        Initializes a FAISS index for vector storage and retrieval.
        
        :param dimension: The dimension of the embeddings (default is 384 for 'all-MiniLM-L6-v2')
        :param use_gpu: Whether to use GPU for FAISS
        """
        
        self.dimension = dimension
        self.use_gpu = use_gpu
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        #Initialize Faiss index
        if self.use_gpu:
            self.res = faiss.StandardGpuResources()
            self.index = faiss.GpuIndexFlatL2(self.res,self.dimension)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            
        self.id_map = {} #Dictionary to map embeddings to original text or identifiers
        
    def add_embeddings(self,texts):
        """
        Adds embeddings for a list of texts to the FAISS index.
        
        :param texts: A list of text strings to embed and add to the index
        """
        embeddings = self.model.encode(texts)
        embeddings = np.array(embeddings).astype('float32') # Convert to float32 for Faiss
        
        self.index.add(embeddings)
        
        #Map IDs to the original text or identifiers
        for idx, text in enumerate(texts):
            self.id_map[self.index.ntotal - len(texts) + idx] = text
            
        print('All extracted data has been added to VectorDB.')
        
    def search(self,query,top_k=5):
        
        """
        Searches the FAISS index for the closest embeddings to the query.
        
        :param query: The query text to search for
        :param top_k: Number of top matches to return
        :return: List of top_k closest results
        """
        
        query_embedding = self.model.encode(query)
        query_embedding = np.array(query_embedding).astype('float32')
         # Ensure query_embedding is a 2D array with shape (1, dimension)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
            distance, indices = self.index.search(query_embedding, top_k)
        
        #Retriuve original text or identifiers based on indices
        result = [(self.id_map[idx],dist) for idx, dist in zip(indices[0],distance[0]) if idx in self.id_map]
        
        return result

    def save_index(self, path="faiss_index.index", id_map_path="id_map.pkl"):
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, path)
        else:
            faiss.write_index(self.index, path)

        # Save id_map using pickle
        with open(id_map_path, 'wb') as f:
            pickle.dump(self.id_map, f)
        
        print(f"Index saved to {path} and id_map saved to {id_map_path}.")


    def load_index(self, path="faiss_index.index", id_map_path="id_map.pkl"):
        cpu_index = faiss.read_index(path)
        if self.use_gpu:
            self.index = faiss.index_cpu_to_gpu(self.res, 0, cpu_index)
        else:
            self.index = cpu_index

        # Load id_map using pickle
        with open(id_map_path, 'rb') as f:
            self.id_map = pickle.load(f)
        
        print(f"Index loaded from {path} and id_map loaded from {id_map_path}.")

