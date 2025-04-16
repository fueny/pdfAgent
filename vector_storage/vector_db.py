"""
Vector Database Module for PDF Analysis System

This module is responsible for embedding text chunks and storing them in a vector database
for similarity search and retrieval.
"""

import os
import time
import pickle
import numpy as np
import faiss
from typing import Dict, List, Any, Optional

# 添加项目根目录到 Python 路径
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from config
import config
from utils.file_utils import ensure_directory_exists


class VectorDatabaseManager:
    """
    Class for managing vector embeddings and database operations.

    Attributes:
        embedding_model: Model for generating text embeddings
        index: FAISS index for vector storage and retrieval
        dimension: Dimension of the embedding vectors
        chunk_metadata: Metadata for each stored chunk
    """

    def __init__(self, model_name: str = None):
        """
        Initialize the vector database manager with an embedding model.

        Args:
            model_name (str, optional): Name of the sentence-transformers model to use.
                If None, uses value from config.
        """
        # Get model name from config or use provided value
        model_name = model_name if model_name is not None else config.VECTOR_MODEL_NAME

        # Import here to avoid potential import issues
        from sentence_transformers import SentenceTransformer

        # Initialize the embedding model
        self.embedding_model = SentenceTransformer(model_name)

        # Get embedding dimension from the model
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()

        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)

        # Storage for chunk metadata
        self.chunk_metadata = []

        # Path for saving/loading the index from config
        self.save_dir = config.VECTOR_DB_PATH
        ensure_directory_exists(self.save_dir)

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text (str): Text to embed

        Returns:
            np.ndarray: Embedding vector
        """
        return self.embedding_model.encode(text)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts (List[str]): Texts to embed

        Returns:
            np.ndarray: Matrix of embedding vectors
        """
        return self.embedding_model.encode(texts)

    def add_texts(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        Add text chunks to the vector database.

        Args:
            chunks (List[Dict[str, Any]]): List of text chunks with metadata

        Returns:
            bool: Success status
        """
        if not chunks:
            return False

        try:
            # Extract texts from chunks
            texts = [chunk["text"] for chunk in chunks]

            # Generate embeddings
            embeddings = self.embed_texts(texts)

            # Convert to float32 for FAISS
            embeddings_np = np.array(embeddings).astype('float32')

            # Add to FAISS index
            self.index.add(embeddings_np)

            # Store metadata
            for chunk in chunks:
                self.chunk_metadata.append({
                    "text": chunk["text"],
                    "metadata": chunk["metadata"]
                })

            return True
        except Exception as e:
            print(f"Error adding texts to vector database: {str(e)}")
            return False

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks to a query.

        Args:
            query (str): Query text
            k (int, optional): Number of results to return, defaults to 5

        Returns:
            List[Dict[str, Any]]: List of similar chunks with metadata and scores
        """
        if not self.chunk_metadata:
            return []

        # Generate query embedding
        query_embedding = self.embed_text(query)
        query_embedding_np = np.array([query_embedding]).astype('float32')

        # Adjust k if it's larger than the number of stored vectors
        k = min(k, len(self.chunk_metadata))

        # Search the index
        distances, indices = self.index.search(query_embedding_np, k)

        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.chunk_metadata):
                continue  # Skip invalid indices

            result = self.chunk_metadata[idx].copy()
            result["score"] = float(distances[0][i])
            results.append(result)

        return results

    def find_similarities(self, chunks: List[Dict[str, Any]], threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Find similarities between chunks from different files.

        Args:
            chunks (List[Dict[str, Any]]): List of text chunks with metadata
            threshold (float, optional): Similarity threshold, defaults to 0.8

        Returns:
            List[Dict[str, Any]]: List of similar chunk pairs
        """
        if not chunks or len(chunks) < 2:
            return []

        # Group chunks by file name
        chunks_by_file = {}
        for chunk in chunks:
            file_name = chunk["metadata"]["file_name"]
            if file_name not in chunks_by_file:
                chunks_by_file[file_name] = []
            chunks_by_file[file_name].append(chunk)

        # If only one file, return empty list
        if len(chunks_by_file) < 2:
            return []

        # Find similarities between chunks from different files
        similarities = []

        # For each file pair
        file_names = list(chunks_by_file.keys())
        for i in range(len(file_names)):
            for j in range(i + 1, len(file_names)):
                file1 = file_names[i]
                file2 = file_names[j]

                # Get chunks for each file
                chunks1 = chunks_by_file[file1]
                chunks2 = chunks_by_file[file2]

                # For each chunk in file1
                for chunk1 in chunks1:
                    # Search for similar chunks in file2
                    similar_chunks = self.search_in_chunks(chunk1["text"], chunks2, threshold)

                    # Add to similarities
                    for similar_chunk in similar_chunks:
                        similarities.append({
                            "chunk1": chunk1,
                            "chunk2": similar_chunk,
                            "file1": file1,
                            "file2": file2,
                            "similarity": similar_chunk["score"]
                        })

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x["similarity"], reverse=True)

        return similarities

    def search_in_chunks(self, query: str, chunks: List[Dict[str, Any]], threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Search for similar chunks in a list of chunks.

        Args:
            query (str): Query text
            chunks (List[Dict[str, Any]]): List of chunks to search in
            threshold (float, optional): Similarity threshold, defaults to 0.8

        Returns:
            List[Dict[str, Any]]: List of similar chunks with scores
        """
        if not chunks:
            return []

        # Extract texts from chunks
        texts = [chunk["text"] for chunk in chunks]

        # Generate query embedding
        query_embedding = self.embed_text(query)

        # Generate embeddings for chunks
        chunk_embeddings = self.embed_texts(texts)

        # Calculate similarities
        similarities = []
        for i, embedding in enumerate(chunk_embeddings):
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))

            # If above threshold, add to results
            if similarity >= threshold:
                result = chunks[i].copy()
                result["score"] = float(similarity)
                similarities.append(result)

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x["score"], reverse=True)

        return similarities

    def save(self, path: str = None) -> bool:
        """
        Save the vector database to disk.

        Args:
            path (str, optional): Path to save to. If None, uses default path.

        Returns:
            bool: Success status
        """
        try:
            # Use provided path or default
            save_path = path or os.path.join(self.save_dir, f"vector_db_{int(time.time())}")

            # Save FAISS index
            faiss.write_index(self.index, f"{save_path}.index")

            # Save metadata
            with open(f"{save_path}.pkl", 'wb') as f:
                pickle.dump(self.chunk_metadata, f)

            return True
        except Exception as e:
            print(f"Error saving vector database: {str(e)}")
            return False

    def load(self, path: str) -> bool:
        """
        Load the vector database from disk.

        Args:
            path (str): Path to load from

        Returns:
            bool: Success status
        """
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{path}.index")

            # Load metadata
            with open(f"{path}.pkl", 'rb') as f:
                self.chunk_metadata = pickle.load(f)

            return True
        except Exception as e:
            print(f"Error loading vector database: {str(e)}")
            return False


# LangGraph node for vector database operations
def vector_db_node(state):
    """
    LangGraph node for embedding and storing text chunks.

    Args:
        state: The current state containing text_chunks

    Returns:
        Updated state with vector_db
    """
    text_chunks = state.get("text_chunks", [])

    if not text_chunks:
        return {
            "status": "error",
            "message": "No text chunks to embed",
            "vector_db": None
        }

    # Create vector database manager
    vector_db = VectorDatabaseManager()

    # Add chunks to vector database
    success = vector_db.add_texts(text_chunks)

    if not success:
        return {
            "status": "error",
            "message": "Failed to add texts to vector database",
            "vector_db": None
        }

    # Find similarities between chunks from different files
    similarities = vector_db.find_similarities(text_chunks)

    return {
        "status": "success",
        "message": f"Successfully embedded {len(text_chunks)} chunks",
        "vector_db": vector_db,
        "similarities": similarities
    }
