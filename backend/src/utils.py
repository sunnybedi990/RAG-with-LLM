import os
import numpy as np

def pad_embedding(embedding, target_dim):
    """
    Pads an embedding to match the target dimension by adding zeros.
    Ensures consistent dimensions before concatenating.
    """
    # Ensure the embedding is a 2D array
    if len(embedding.shape) == 1:
        embedding = embedding.reshape(1, -1)  # Reshape to (1, embedding_dim)
    
    # Calculate the padding size
    current_dim = embedding.shape[-1]
    if current_dim < target_dim:
        padding = np.zeros((embedding.shape[0], target_dim - current_dim))  # Match batch size
        return np.hstack((embedding, padding))
    return embedding

def extract_name_from_path(db_path):
    """Extract the collection or index name from the database path."""
    return os.path.splitext(os.path.basename(db_path))[0]
