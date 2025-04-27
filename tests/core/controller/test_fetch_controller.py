from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.qa_gpt.core.controller.fetch_controller import FetchController
from src.qa_gpt.core.objects.materials import FileMeta


@pytest.fixture
def fetch_controller():
    return FetchController()


@pytest.fixture
def mock_material_controller():
    return MagicMock()


@pytest.fixture
def mock_rag_controller():
    return MagicMock()


@pytest.fixture
def sample_file_meta():
    return FileMeta(
        id=1,
        file_name="test.pdf",
        file_suffix=".pdf",
        file_path=Path("test.pdf"),
        mc_question_sets={},
        summaries={},
        parsing_results={
            "sections": [{"content": "Section 1 content"}, {"content": "Section 2 content"}],
            "images": [],
            "tables": [],
        },
        rag_state=None,
    )


@pytest.mark.asyncio
async def test_build_rag_index_single_file(
    fetch_controller, mock_material_controller, mock_rag_controller, sample_file_meta
):
    # Setup
    fetch_controller.material_controller = mock_material_controller
    mock_material_controller.get_material_table.return_value = {"file1": sample_file_meta}

    # Mock RAGController
    with patch("src.qa_gpt.core.controller.fetch_controller.RAGController") as mock_rag_class:
        mock_rag_class.return_value = mock_rag_controller

        # Execute
        await fetch_controller.build_rag_index(file_id="file1")

        # Verify
        # Check that RAG state folder was created
        assert Path("./rag_state").exists()

        # Check that RAG controller was initialized with correct path
        mock_rag_class.assert_called_once()
        assert "file1_rag_state.json" in str(mock_rag_class.call_args[1]["index_path"])

        # Check that texts were added to RAG index
        mock_rag_controller.add_texts.assert_called_once_with(
            ["Section 1 content", "Section 2 content"]
        )

        # Check that RAG state was saved
        mock_rag_controller.save_state.assert_called_once()

        # Check that file meta was updated and saved
        assert sample_file_meta.rag_state is not None
        mock_material_controller.db_controller.save_data.assert_called_once()


@pytest.mark.asyncio
async def test_build_rag_index_skip_existing(
    fetch_controller, mock_material_controller, sample_file_meta, tmp_path
):
    # Setup
    fetch_controller.material_controller = mock_material_controller
    rag_state_path = tmp_path / "rag_state" / "file1_rag_state.json"
    rag_state_path.parent.mkdir(exist_ok=True)
    rag_state_path.touch()
    sample_file_meta.rag_state = rag_state_path
    mock_material_controller.get_material_table.return_value = {"file1": sample_file_meta}

    # Execute
    await fetch_controller.build_rag_index(file_id="file1")

    # Mock RAGController to verify it was not called
    with patch("src.qa_gpt.core.controller.fetch_controller.RAGController") as mock_rag_class:
        mock_rag_class.assert_not_called()


@pytest.mark.asyncio
async def test_build_rag_index_skip_no_parsing(
    fetch_controller, mock_material_controller, sample_file_meta, tmp_path
):
    # Setup
    fetch_controller.material_controller = mock_material_controller
    sample_file_meta.parsing_results["sections"] = None
    mock_material_controller.get_material_table.return_value = {"file1": sample_file_meta}

    # Execute
    await fetch_controller.build_rag_index(file_id="file1")

    # Mock RAGController to verify it was not called
    with patch("src.qa_gpt.core.controller.fetch_controller.RAGController") as mock_rag_class:
        mock_rag_class.assert_not_called()


@pytest.mark.asyncio
async def test_build_rag_index_process_all(
    fetch_controller, mock_material_controller, mock_rag_controller, sample_file_meta
):
    # Setup
    fetch_controller.material_controller = mock_material_controller
    # Create a second file meta with different ID but same content
    sample_file_meta2 = FileMeta(
        id=2,
        file_name="test2.pdf",
        file_suffix=".pdf",
        file_path=Path("test2.pdf"),
        mc_question_sets={},
        summaries={},
        parsing_results={
            "sections": [{"content": "Section 1 content"}, {"content": "Section 2 content"}],
            "images": [],
            "tables": [],
        },
        rag_state=None,
    )
    mock_material_controller.get_material_table.return_value = {
        "file1": sample_file_meta,
        "file2": sample_file_meta2,
    }

    # Mock RAGController
    with patch("src.qa_gpt.core.controller.fetch_controller.RAGController") as mock_rag_class:
        mock_rag_class.return_value = mock_rag_controller

        # Execute
        await fetch_controller.build_rag_index(process_all=True)

        # Verify
        # Check that RAG controller was called twice (once for each file)
        assert mock_rag_class.call_count == 2
        assert mock_rag_controller.add_texts.call_count == 2
        assert mock_rag_controller.save_state.call_count == 2


@pytest.mark.asyncio
async def test_build_rag_index_error_handling(
    fetch_controller, mock_material_controller, sample_file_meta
):
    # Setup
    fetch_controller.material_controller = mock_material_controller
    mock_material_controller.get_material_table.return_value = {"file1": sample_file_meta}

    # Mock RAGController to raise an exception
    with patch("src.qa_gpt.core.controller.fetch_controller.RAGController") as mock_rag_class:
        mock_rag_class.side_effect = Exception("Test error")

        # Execute and verify no exception is raised
        await fetch_controller.build_rag_index(file_id="file1")

        # Verify that the error was handled gracefully
        assert not Path("./rag_state/file1_rag_state.json").exists()
