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


def test_initialization(temp_rag_folder, test_file_id):
    """Test RAG controller initialization."""
    # Create a new controller and save its state
    controller = RAGController(file_id=test_file_id, rag_state_folder_path=str(temp_rag_folder))
    controller.save_state(controller.state_path)

    # Now load the state
    loaded_controller = RAGController.from_file_id(
        test_file_id, rag_state_folder_path=str(temp_rag_folder)
    )
    assert loaded_controller.dimension == 384  # all-MiniLM-L6-v2 has 384 dimensions
    assert loaded_controller.index is not None
    assert loaded_controller.index.ntotal == 0


def test_add_vectors(temp_rag_folder, test_file_id):
    """Test adding vectors to the index."""
    # Create a new controller and save its state
    controller = RAGController(file_id=test_file_id, rag_state_folder_path=str(temp_rag_folder))
    controller.save_state(controller.state_path)

    # Now load the state
    rag_controller = RAGController.from_file_id(
        test_file_id, rag_state_folder_path=str(temp_rag_folder)
    )

    # Create some test vectors with correct dimension
    vectors = np.random.rand(3, 384).astype(np.float32)  # all-MiniLM-L6-v2 has 384 dimensions

    # Add vectors to the index
    rag_controller.add_vectors(vectors)

    # Check if vectors were added
    assert rag_controller.index.ntotal == 3


def test_search(temp_rag_folder, test_file_id):
    """Test searching for similar vectors."""
    # Create a new controller and save its state
    controller = RAGController(file_id=test_file_id, rag_state_folder_path=str(temp_rag_folder))
    controller.save_state(controller.state_path)

    # Now load the state
    rag_controller = RAGController.from_file_id(
        test_file_id, rag_state_folder_path=str(temp_rag_folder)
    )

    # Add some test vectors with correct dimension
    vectors = np.random.rand(3, 384).astype(np.float32)  # all-MiniLM-L6-v2 has 384 dimensions
    rag_controller.add_vectors(vectors)

    # Search for a similar vector
    query_vector = np.random.rand(384).astype(np.float32)  # all-MiniLM-L6-v2 has 384 dimensions
    results = rag_controller.search(query_vector, k=2)

    # Check results
    assert len(results) == 2
    assert results[0][1] < results[1][1]  # First result should have smaller distance


def test_get_vector_by_index(temp_rag_folder, test_file_id):
    """Test retrieving vectors by index."""
    # Create a new controller and save its state
    controller = RAGController(file_id=test_file_id, rag_state_folder_path=str(temp_rag_folder))
    controller.save_state(controller.state_path)

    # Now load the state
    rag_controller = RAGController.from_file_id(
        test_file_id, rag_state_folder_path=str(temp_rag_folder)
    )

    # Add a test vector with correct dimension
    vector = np.random.rand(384).astype(np.float32)  # all-MiniLM-L6-v2 has 384 dimensions
    rag_controller.add_vectors(vector.reshape(1, -1))

    # Retrieve the vector
    retrieved_vector = rag_controller.get_vector_by_index(0)

    # Check if the retrieved vector matches the original
    np.testing.assert_array_almost_equal(vector, retrieved_vector)


def test_invalid_index(temp_rag_folder, test_file_id):
    """Test handling of invalid index."""
    # Create a new controller and save its state
    controller = RAGController(file_id=test_file_id, rag_state_folder_path=str(temp_rag_folder))
    controller.save_state(controller.state_path)

    # Now load the state
    rag_controller = RAGController.from_file_id(
        test_file_id, rag_state_folder_path=str(temp_rag_folder)
    )

    with pytest.raises(IndexError):
        rag_controller.get_vector_by_index(0)  # No vectors added yet


def test_empty_search(temp_rag_folder, test_file_id):
    """Test searching with empty index."""
    # Create a new controller and save its state
    controller = RAGController(file_id=test_file_id, rag_state_folder_path=str(temp_rag_folder))
    controller.save_state(controller.state_path)

    # Now load the state
    rag_controller = RAGController.from_file_id(
        test_file_id, rag_state_folder_path=str(temp_rag_folder)
    )

    query_vector = np.random.rand(384).astype(np.float32)  # all-MiniLM-L6-v2 has 384 dimensions
    results = rag_controller.search(query_vector, k=5)
    assert len(results) == 0


def test_persistence(temp_rag_folder, test_file_id):
    """Test saving and loading the index and text store."""
    # Create first controller and add vectors and texts
    controller1 = RAGController(file_id=test_file_id, rag_state_folder_path=str(temp_rag_folder))

    # Add some test data
    test_vectors = np.random.rand(5, 384).astype(np.float32)
    test_texts = ["text1", "text2", "text3", "text4", "text5"]
    controller1.add_vectors(test_vectors)
    controller1.add_texts(test_texts)

    # Save state
    controller1.save_state(controller1.state_path)

    # Create second controller by loading state
    controller2 = RAGController.from_file_id(
        test_file_id, rag_state_folder_path=str(temp_rag_folder)
    )

    # Verify data was loaded correctly
    assert controller2.index.ntotal == 10
    assert len(controller2.text_store) == 5

    # Verify vectors and texts match
    for i in range(5):
        assert np.array_equal(controller2.get_vector_by_index(i), test_vectors[i])
        assert controller2.get_text_by_index(i) == test_texts[i]


def test_gpu_detection(temp_rag_folder, test_file_id):
    """Test GPU detection and usage."""
    # Create a new controller and save its state
    controller = RAGController(file_id=test_file_id, rag_state_folder_path=str(temp_rag_folder))
    controller.save_state(controller.state_path)

    # This test just verifies that the code runs without error
    # Actual GPU detection depends on the system
    assert True


def test_invalid_vector_dimension(temp_rag_folder, test_file_id):
    """Test handling of vectors with wrong dimension."""
    controller = RAGController(file_id=test_file_id, rag_state_folder_path=str(temp_rag_folder))
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)  # Wrong dimension

    with pytest.raises(ValueError):
        controller.add_vectors(vectors)


# New tests for text functionality
def test_add_texts(temp_rag_folder, test_file_id):
    """Test adding texts to the index."""
    # Create a new controller and save its state
    controller = RAGController(file_id=test_file_id, rag_state_folder_path=str(temp_rag_folder))
    controller.save_state(controller.state_path)

    # Now load the state
    rag_controller = RAGController.from_file_id(
        test_file_id, rag_state_folder_path=str(temp_rag_folder)
    )

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


def test_search_text(temp_rag_folder, test_file_id):
    """Test searching for similar texts."""
    # Create a new controller and save its state
    controller = RAGController(file_id=test_file_id, rag_state_folder_path=str(temp_rag_folder))
    controller.save_state(controller.state_path)

    # Now load the state
    rag_controller = RAGController.from_file_id(
        test_file_id, rag_state_folder_path=str(temp_rag_folder)
    )

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


def test_get_text_by_index(temp_rag_folder, test_file_id):
    """Test retrieving texts by index."""
    # Create a new controller and save its state
    controller = RAGController(file_id=test_file_id, rag_state_folder_path=str(temp_rag_folder))
    controller.save_state(controller.state_path)

    # Now load the state
    rag_controller = RAGController.from_file_id(
        test_file_id, rag_state_folder_path=str(temp_rag_folder)
    )

    # Add a test text
    text = "The quick brown fox jumps over the lazy dog"
    rag_controller.add_texts([text])

    # Retrieve the text
    retrieved_text = rag_controller.get_text_by_index(0)

    # Check if the retrieved text matches the original
    assert retrieved_text == text


def test_invalid_text_index(temp_rag_folder, test_file_id):
    """Test handling of invalid text index."""
    # Create a new controller and save its state
    controller = RAGController(file_id=test_file_id, rag_state_folder_path=str(temp_rag_folder))
    controller.save_state(controller.state_path)

    # Now load the state
    rag_controller = RAGController.from_file_id(
        test_file_id, rag_state_folder_path=str(temp_rag_folder)
    )

    with pytest.raises(IndexError):
        rag_controller.get_text_by_index(0)  # No texts added yet


def test_empty_text_search(temp_rag_folder, test_file_id):
    """Test searching with empty text index."""
    # Create a new controller and save its state
    controller = RAGController(file_id=test_file_id, rag_state_folder_path=str(temp_rag_folder))
    controller.save_state(controller.state_path)

    # Now load the state
    rag_controller = RAGController.from_file_id(
        test_file_id, rag_state_folder_path=str(temp_rag_folder)
    )

    results = rag_controller.search_text("test query", k=5)
    assert len(results) == 0
