import numpy as np
import faiss
import pickle
import os

class FAISSVectorDB:
    def __init__(self, use_gpu=True, dimension=768):
        self.use_gpu = use_gpu
        self.id_map = {}
        self.dimension = dimension  # Default dimension (adjust based on your embeddings)

        # Initialize FAISS index
        if use_gpu:
            self.res = faiss.StandardGpuResources()
            self.index = faiss.GpuIndexFlatL2(self.res, self.dimension)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)

    def add_embeddings(self, embeddings, texts):
        self.index.add(embeddings)
        self.update_id_map(texts)

    def search(self, query_embedding, top_k=5):
        """
        Searches the FAISS index for the closest embeddings.
        Returns a list of (text, score) tuples.
        """
        print('faiss')
        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        results = [(self.id_map[idx], distances[0][i]) for i, idx in enumerate(indices[0]) if idx in self.id_map]
        return results


    def save_index(self, path):
            """
            Saves the FAISS index to disk. Converts GPU index to CPU index if necessary.
            """
            try:
                id_map_path = os.path.splitext(path)[0] + ".pkl"

                if self.use_gpu:
                    # Convert GPU index to CPU index before saving
                    cpu_index = faiss.index_gpu_to_cpu(self.index)
                    faiss.write_index(cpu_index, path)
                    with open(id_map_path, 'wb') as f:
                     pickle.dump(self.id_map, f)
#             print(f"Index saved to {path}, ID map saved to {id_map_path}")
                else:
                    faiss.write_index(self.index, path)
                    with open(id_map_path, 'wb') as f:
                     pickle.dump(self.id_map, f)
#             print(f"Index saved to {path}, ID map saved to {id_map_path}")
                print(f"FAISS index saved to {path}.")
            except Exception as e:
                raise RuntimeError(f"Error saving FAISS index: {e}")

    def load_index(self, path):
            """
            Loads the FAISS index from disk. Converts it to GPU index if necessary.
            """
            try:
                id_map_path = os.path.splitext(path)[0] + ".pkl"

                cpu_index = faiss.read_index(path)
                if self.use_gpu:
                    self.index = faiss.index_cpu_to_gpu(self.res, 0, cpu_index)
                    with open(id_map_path, 'rb') as f:
                        self.id_map = pickle.load(f)
                    print(f"Index loaded from {path}, ID map loaded from {id_map_path}")
                else:
                    self.index = cpu_index
                    with open(id_map_path, 'rb') as f:
                        self.id_map = pickle.load(f)
                    print(f"Index loaded from {path}, ID map loaded from {id_map_path}")
                print(f"FAISS index loaded from {path}.")
            except Exception as e:
                raise RuntimeError(f"Error loading FAISS index: {e}")

    def update_id_map(self, texts):
        start_id = self.index.ntotal - len(texts)
        for i, text in enumerate(texts):
            self.id_map[start_id + i] = text
    
    def get_all(self):
        """
        Retrieve all embeddings and their associated texts from the FAISS index.
        """
        total_vectors = self.index.ntotal
        if total_vectors == 0:
            return []
        embeddings = [self.index.reconstruct(i) for i in range(total_vectors)]
        results = [self.id_map[idx] for idx in total_vectors if idx in self.id_map]
        return results

