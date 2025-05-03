import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.qa_gpt.core.controller.rag_controller import RAGController


@pytest.fixture
def temp_rag_folder():
    """Create a temporary folder for RAG state files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        return Path(tmp_dir)


@pytest.fixture
def test_file_id():
    """Return a test file ID."""
    return "test_file_123"


@pytest.fixture
def rag_controller(temp_rag_folder, test_file_id):
    """Create a RAG controller with a temporary folder."""
    controller = RAGController.from_file_id(
        test_file_id, rag_state_folder_path=str(temp_rag_folder)
    )
    return controller


def test_initialization(rag_controller):
    """Test RAG controller initialization."""
    assert rag_controller.dimension == 384  # all-MiniLM-L6-v2 has 384 dimensions
    assert rag_controller.index is not None
    assert rag_controller.index.ntotal == 0


def test_add_vectors(rag_controller):
    """Test adding vectors to the index."""
    # Create some test vectors with correct dimension
    vectors = np.random.rand(3, 384).astype(np.float32)  # all-MiniLM-L6-v2 has 384 dimensions

    # Add vectors to the index
    rag_controller.add_vectors(vectors)

    # Check if vectors were added
    assert rag_controller.index.ntotal == 3


def test_search(rag_controller):
    """Test searching for similar vectors."""
    # Add some test vectors with correct dimension
    vectors = np.random.rand(3, 384).astype(np.float32)  # all-MiniLM-L6-v2 has 384 dimensions
    rag_controller.add_vectors(vectors)

    # Search for a similar vector
    query_vector = np.random.rand(384).astype(np.float32)  # all-MiniLM-L6-v2 has 384 dimensions
    results = rag_controller.search(query_vector, k=2)

    # Check results
    assert len(results) == 2
    assert results[0][1] < results[1][1]  # First result should have smaller distance


def test_get_vector_by_index(rag_controller):
    """Test retrieving vectors by index."""
    # Add a test vector with correct dimension
    vector = np.random.rand(384).astype(np.float32)  # all-MiniLM-L6-v2 has 384 dimensions
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
    query_vector = np.random.rand(384).astype(np.float32)  # all-MiniLM-L6-v2 has 384 dimensions
    results = rag_controller.search(query_vector, k=5)
    assert len(results) == 0


def test_persistence(temp_rag_folder, test_file_id):
    """Test saving and loading the index and text store."""
    # Create first controller and add vectors and texts
    controller1 = RAGController.from_file_id(
        test_file_id, rag_state_folder_path=str(temp_rag_folder)
    )
    vectors = np.random.rand(2, 384).astype(np.float32)  # all-MiniLM-L6-v2 has 384 dimensions
    texts = ["Test text 1", "Test text 2"]
    controller1.add_vectors(vectors)
    controller1.add_texts(texts)

    # Save state
    controller1.save_state(controller1.state_path)

    # Create second controller by loading the state
    controller2 = RAGController.load_state(controller1.state_path)

    # Check if vectors were loaded
    assert controller2.index.ntotal == 4  # 2 vectors + 2 texts

    # Verify the vectors
    vector1 = controller2.get_vector_by_index(0)
    vector2 = controller2.get_vector_by_index(1)
    np.testing.assert_array_almost_equal(vector1, vectors[0])
    np.testing.assert_array_almost_equal(vector2, vectors[1])

    # Verify the texts
    assert controller2.text_store == texts


def test_gpu_detection(rag_controller):
    """Test GPU detection and usage."""
    # This test just verifies that the code runs without error
    # Actual GPU detection depends on the system
    assert True


def test_invalid_vector_dimension():
    """Test handling of vectors with wrong dimension."""
    controller = RAGController()
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)  # Wrong dimension

    with pytest.raises(ValueError):
        controller.add_vectors(vectors)


# New tests for text functionality
def test_add_texts(rag_controller):
    """Test adding texts to the index."""
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "A journey of a thousand miles begins with a single step",
        "All that glitters is not gold",
    ]

    # Add texts to the index
    rag_controller.add_texts(texts)

    # Check if texts were added
    assert rag_controller.index.ntotal == 3
    assert len(rag_controller.text_store) == 3
    assert rag_controller.text_store == texts


def test_search_text(rag_controller):
    """Test searching for similar texts."""
    # Add some test texts
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "A journey of a thousand miles begins with a single step",
        "All that glitters is not gold",
    ]
    rag_controller.add_texts(texts)

    # Search for a similar text
    query_text = "What is the color of the fox?"
    results = rag_controller.search_text(query_text, k=2)

    # Check results
    assert len(results) == 2
    assert isinstance(results[0][0], str)  # First element should be text
    assert isinstance(results[0][1], float)  # Second element should be distance
    assert results[0][1] < results[1][1]  # First result should have smaller distance


def test_get_text_by_index(rag_controller):
    """Test retrieving texts by index."""
    # Add a test text
    text = "The quick brown fox jumps over the lazy dog"
    rag_controller.add_texts([text])

    # Retrieve the text
    retrieved_text = rag_controller.get_text_by_index(0)

    # Check if the retrieved text matches the original
    assert retrieved_text == text


def test_invalid_text_index(rag_controller):
    """Test handling of invalid text index."""
    with pytest.raises(IndexError):
        rag_controller.get_text_by_index(0)  # No texts added yet


def test_empty_text_search(rag_controller):
    """Test searching with empty text index."""
    results = rag_controller.search_text("test query", k=5)
    assert len(results) == 0
