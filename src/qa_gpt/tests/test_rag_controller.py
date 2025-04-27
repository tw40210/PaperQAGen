import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.qa_gpt.core.controller.rag_controller import RAGController


@pytest.fixture
def temp_index_path():
    """Create a temporary file for the FAISS index."""
    with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as tmp:
        return Path(tmp.name)


@pytest.fixture
def rag_controller(temp_index_path):
    """Create a RAG controller with a temporary index path."""
    return RAGController(dimension=4, index_path=temp_index_path)


def test_initialization(rag_controller):
    """Test RAG controller initialization."""
    assert rag_controller.dimension == 4
    assert rag_controller.index is not None
    assert rag_controller.index.ntotal == 0


def test_add_vectors(rag_controller):
    """Test adding vectors to the index."""
    # Create some test vectors
    vectors = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float32
    )

    # Add vectors to the index
    rag_controller.add_vectors(vectors)

    # Check if vectors were added
    assert rag_controller.index.ntotal == 3


def test_search(rag_controller):
    """Test searching for similar vectors."""
    # Add some test vectors
    vectors = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float32
    )
    rag_controller.add_vectors(vectors)

    # Search for a similar vector
    query_vector = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
    results = rag_controller.search(query_vector, k=2)

    # Check results
    assert len(results) == 2
    assert results[0][0] == 0  # Should match first vector
    assert results[0][1] < results[1][1]  # First result should have smaller distance


def test_get_vector_by_index(rag_controller):
    """Test retrieving vectors by index."""
    # Add a test vector
    vector = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    rag_controller.add_vectors(vector.reshape(1, -1))

    # Retrieve the vector
    retrieved_vector = rag_controller.get_vector_by_index(0)

    # Check if the retrieved vector matches the original
    np.testing.assert_array_almost_equal(vector, retrieved_vector)


def test_invalid_index(rag_controller):
    """Test handling of invalid index."""
    with pytest.raises(IndexError):
        rag_controller.get_vector_by_index(0)  # No vectors added yet


def test_empty_search(rag_controller):
    """Test searching with empty index."""
    query_vector = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    results = rag_controller.search(query_vector, k=5)
    assert len(results) == 0


def test_persistence(temp_index_path):
    """Test saving and loading the index."""
    # Create first controller and add vectors
    controller1 = RAGController(dimension=4, index_path=temp_index_path)
    vectors = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    controller1.add_vectors(vectors)

    # Create second controller and load the index
    controller2 = RAGController(dimension=4, index_path=temp_index_path)

    # Check if vectors were loaded
    assert controller2.index.ntotal == 2

    # Verify the vectors
    vector1 = controller2.get_vector_by_index(0)
    vector2 = controller2.get_vector_by_index(1)
    np.testing.assert_array_almost_equal(vector1, vectors[0])
    np.testing.assert_array_almost_equal(vector2, vectors[1])


def test_invalid_vector_dimension():
    """Test handling of vectors with wrong dimension."""
    controller = RAGController(dimension=4)
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)  # Wrong dimension

    with pytest.raises(ValueError):
        controller.add_vectors(vectors)


def test_cleanup(temp_index_path):
    """Clean up the temporary index file."""
    if temp_index_path.exists():
        os.unlink(temp_index_path)
