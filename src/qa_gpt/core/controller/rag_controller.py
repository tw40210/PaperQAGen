import logging
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class RAGController:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", index_path: Path = None):
        """
        Initialize the RAG controller with FAISS index.

        Args:
            model_name: Name of the sentence transformer model to use for text embeddings
            index_path: Path to save/load the FAISS index. If None, index will be kept in memory
        """
        self.index_path = index_path
        self.index = None
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.text_store = []  # Store original texts

        if index_path and index_path.exists():
            self._load_index()
        else:
            self._init_index()

    def _init_index(self):
        """Initialize a new FAISS index."""
        self.index = faiss.IndexFlatL2(self.dimension)
        if faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)

    def _load_index(self):
        """Load an existing FAISS index from disk."""
        try:
            self.index = faiss.read_index(str(self.index_path))
            if faiss.get_num_gpus() > 0:
                self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
            logger.info(f"Successfully loaded FAISS index from {self.index_path}")
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            self._init_index()

    def _save_index(self):
        """Save the FAISS index to disk."""
        if self.index_path:
            try:
                cpu_index = (
                    faiss.index_gpu_to_cpu(self.index) if faiss.get_num_gpus() > 0 else self.index
                )
                faiss.write_index(cpu_index, str(self.index_path))
                logger.info(f"Successfully saved FAISS index to {self.index_path}")
            except Exception as e:
                logger.error(f"Failed to save index: {e}")

    def add_texts(self, texts: list[str]) -> None:
        """
        Add a list of texts to the vector store.

        Args:
            texts: List of strings to be embedded and stored
        """
        if not texts:
            return

        # Generate embeddings
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        embeddings = embeddings.cpu().numpy().astype("float32")

        # Add to FAISS index
        self.index.add(embeddings)

        # Store original texts
        self.text_store.extend(texts)

        # Save index if path is specified
        self._save_index()

    def add_vectors(self, vectors: np.ndarray) -> None:
        """
        Add vectors to the vector store.

        Args:
            vectors: numpy array of shape (n, dimension) containing vectors to be stored

        Raises:
            ValueError: If vectors have wrong dimension
        """
        if vectors.size == 0:
            return

        # Ensure vectors are float32
        vectors = vectors.astype("float32")

        # Check vector dimensions
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected vectors of dimension {self.dimension}, got {vectors.shape[1]}"
            )

        # Add to FAISS index
        self.index.add(vectors)

        # Save index if path is specified
        self._save_index()

    def search_text(self, query_text: str, k: int = 5) -> list[tuple[str, float]]:
        """
        Search for the most relevant texts given a query text.

        Args:
            query_text: The search query string
            k: Number of results to return

        Returns:
            List of tuples containing (text, distance) for the top k results
        """
        if not self.text_store:
            return []

        # Generate query embedding
        query_embedding = self.model.encode(query_text, convert_to_tensor=True)
        query_embedding = query_embedding.cpu().numpy().astype("float32")
        query_embedding = np.expand_dims(query_embedding, axis=0)

        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, k)

        # Get results
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= 0 and idx < len(self.text_store):  # FAISS returns -1 for empty results
                results.append((self.text_store[idx], float(dist)))

        return results

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
        """
        Search for the most relevant vectors given a query vector.

        Args:
            query_vector: numpy array of shape (dimension,) containing the query vector
            k: Number of results to return

        Returns:
            List of tuples containing (index, distance) for the top k results

        Raises:
            ValueError: If query vector has wrong dimension
        """
        if self.index.ntotal == 0:
            return []

        # Ensure query vector is float32 and has correct shape
        query_vector = query_vector.astype("float32")

        # Check vector dimension
        if query_vector.shape[0] != self.dimension:
            raise ValueError(
                f"Expected query vector of dimension {self.dimension}, got {query_vector.shape[0]}"
            )

        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)

        # Search in FAISS index
        distances, indices = self.index.search(query_vector, k)

        # Get results
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= 0:  # FAISS returns -1 for empty results
                results.append((int(idx), float(dist)))

        return results

    def get_vector_by_index(self, index: int) -> np.ndarray:
        """
        Retrieve the vector by its index in the store.

        Args:
            index: Index of the vector to retrieve

        Returns:
            The vector as a numpy array

        Raises:
            IndexError: If index is out of bounds
        """
        if not 0 <= index < self.index.ntotal:
            raise IndexError(
                f"Index {index} is out of bounds for vector store of size {self.index.ntotal}"
            )

        return self.index.reconstruct(index)

    def get_text_by_index(self, index: int) -> str:
        """
        Retrieve the text by its index in the store.

        Args:
            index: Index of the text to retrieve

        Returns:
            The text string

        Raises:
            IndexError: If index is out of bounds
        """
        if not 0 <= index < len(self.text_store):
            raise IndexError(
                f"Index {index} is out of bounds for text store of size {len(self.text_store)}"
            )

        return self.text_store[index]
