import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from src.qa_gpt.core.objects.questions import QuestionComment
from src.qa_gpt.core.ui.materials_overview import get_all_materials


@pytest.fixture
def temp_folder():
    """Create a temporary folder for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_material_controller():
    """Create a mock MaterialController."""
    controller = MagicMock()
    return controller


@pytest.fixture
def sample_question_comment():
    """Create a sample QuestionComment."""
    return QuestionComment(
        topic="difficulty",
        content="This question is too complex",
        is_positive=False,
        question_set_id="StandardSummary_bullet_points_0",
        question_id="question_2",
    )


@pytest.fixture
def sample_positive_comment():
    """Create a sample positive QuestionComment."""
    return QuestionComment(
        topic="clarity",
        content="This question is very clear",
        is_positive=True,
        question_set_id="StandardSummary_bullet_points_0",
        question_id="question_1",
    )


@pytest.fixture
def mock_file_meta_with_comments(sample_question_comment, sample_positive_comment):
    """Create a mock FileMeta with comments."""
    file_meta = MagicMock()
    file_meta.question_comments = {
        "StandardSummary_bullet_points_0_question_2_0": sample_question_comment,
        "StandardSummary_bullet_points_0_question_1_0": sample_positive_comment,
    }
    return file_meta


@pytest.fixture
def mock_file_meta_no_comments():
    """Create a mock FileMeta without comments."""
    file_meta = MagicMock()
    file_meta.question_comments = {}
    return file_meta


def create_test_json_file(folder_path: str, filename: str, content: dict = None):
    """Helper function to create a test JSON file."""
    if content is None:
        content = {"test": "data"}

    file_path = os.path.join(folder_path, filename)
    with open(file_path, "w") as f:
        json.dump(content, f)
    return file_path


def test_get_all_materials_empty_folder(temp_folder):
    """Test get_all_materials with an empty folder."""
    with patch("src.qa_gpt.core.ui.materials_overview.initialize_controllers") as mock_init:
        mock_controller = MagicMock()
        mock_init.return_value = mock_controller
        mock_controller.get_material_table.return_value = {}

        result = get_all_materials(temp_folder)

        assert result == []
        mock_init.assert_called_once()


def test_get_all_materials_no_json_files(temp_folder):
    """Test get_all_materials with folder containing no JSON files."""
    # Create a non-JSON file
    create_test_json_file(temp_folder, "test.txt", {"test": "data"})

    with patch("src.qa_gpt.core.ui.materials_overview.initialize_controllers") as mock_init:
        mock_controller = MagicMock()
        mock_init.return_value = mock_controller
        mock_controller.get_material_table.return_value = {}

        result = get_all_materials(temp_folder)

        assert result == []
        mock_init.assert_called_once()


def test_get_all_materials_single_json_file(temp_folder, mock_file_meta_no_comments):
    """Test get_all_materials with a single JSON file."""
    # Create a test JSON file
    filename = "mc_question_StandardSummary_bullet_points_0.json"
    create_test_json_file(temp_folder, filename)

    with patch("src.qa_gpt.core.ui.materials_overview.initialize_controllers") as mock_init:
        mock_controller = MagicMock()
        mock_init.return_value = mock_controller
        mock_controller.get_material_table.return_value = {}
        mock_controller.get_material_by_filename.return_value = mock_file_meta_no_comments

        result = get_all_materials(temp_folder)

        assert len(result) == 1
        material = result[0]
        assert material["question_set_name"] == "StandardSummary_bullet_points_0"
        assert material["file"] == filename
        assert material["material_name"] == "."  # os.path.relpath returns "." for current directory
        assert material["total_comments"] == 0
        assert material["positive_comments"] == 0
        assert material["full_path"] == os.path.join(temp_folder, filename)


def test_get_all_materials_with_comments(temp_folder, mock_file_meta_with_comments):
    """Test get_all_materials with materials that have comments."""
    # Create a test JSON file
    filename = "mc_question_StandardSummary_bullet_points_0.json"
    create_test_json_file(temp_folder, filename)

    with patch("src.qa_gpt.core.ui.materials_overview.initialize_controllers") as mock_init:
        mock_controller = MagicMock()
        mock_init.return_value = mock_controller
        mock_controller.get_material_table.return_value = {}
        mock_controller.get_material_by_filename.return_value = mock_file_meta_with_comments

        result = get_all_materials(temp_folder)

        assert len(result) == 1
        material = result[0]
        assert material["question_set_name"] == "StandardSummary_bullet_points_0"
        assert material["total_comments"] == 2
        assert material["positive_comments"] == 1


def test_get_all_materials_multiple_files(temp_folder, mock_file_meta_no_comments):
    """Test get_all_materials with multiple JSON files."""
    # Create multiple test JSON files
    files = [
        "mc_question_StandardSummary_bullet_points_0.json",
        "mc_question_TechnicalSummary_overview_0.json",
        "mc_question_InnovationSummary_key_concepts_0.json",
    ]

    for filename in files:
        create_test_json_file(temp_folder, filename)

    with patch("src.qa_gpt.core.ui.materials_overview.initialize_controllers") as mock_init:
        mock_controller = MagicMock()
        mock_init.return_value = mock_controller
        mock_controller.get_material_table.return_value = {}
        mock_controller.get_material_by_filename.return_value = mock_file_meta_no_comments

        result = get_all_materials(temp_folder)

        assert len(result) == 3
        names = [material["question_set_name"] for material in result]
        assert "StandardSummary_bullet_points_0" in names
        assert "TechnicalSummary_overview_0" in names
        assert "InnovationSummary_key_concepts_0" in names


def test_get_all_materials_nested_folders(temp_folder, mock_file_meta_no_comments):
    """Test get_all_materials with nested folder structure."""
    # Create nested folder structure
    nested_folder = os.path.join(temp_folder, "Mamba_0")
    os.makedirs(nested_folder, exist_ok=True)

    filename = "mc_question_StandardSummary_bullet_points_0.json"
    create_test_json_file(nested_folder, filename)

    with patch("src.qa_gpt.core.ui.materials_overview.initialize_controllers") as mock_init:
        mock_controller = MagicMock()
        mock_init.return_value = mock_controller
        mock_controller.get_material_table.return_value = {}
        mock_controller.get_material_by_filename.return_value = mock_file_meta_no_comments

        result = get_all_materials(temp_folder)

        assert len(result) == 1
        material = result[0]
        assert material["question_set_name"] == "StandardSummary_bullet_points_0"
        assert (
            material["material_name"] == "Mamba"
        )  # Implementation splits on "_" and takes first part
        assert material["file"] == filename


def test_get_all_materials_file_meta_not_found(temp_folder):
    """Test get_all_materials when file metadata is not found in database."""
    filename = "mc_question_StandardSummary_bullet_points_0.json"
    create_test_json_file(temp_folder, filename)

    with patch("src.qa_gpt.core.ui.materials_overview.initialize_controllers") as mock_init:
        mock_controller = MagicMock()
        mock_init.return_value = mock_controller
        mock_controller.get_material_table.return_value = {}
        mock_controller.get_material_by_filename.return_value = None

        result = get_all_materials(temp_folder)

        assert len(result) == 1
        material = result[0]
        assert material["question_set_name"] == "StandardSummary_bullet_points_0"
        assert material["total_comments"] == 0
        assert material["positive_comments"] == 0


def test_get_all_materials_mixed_file_types(temp_folder, mock_file_meta_no_comments):
    """Test get_all_materials with mixed file types (JSON and non-JSON)."""
    # Create JSON and non-JSON files
    json_files = [
        "mc_question_StandardSummary_bullet_points_0.json",
        "mc_question_TechnicalSummary_overview_0.json",
    ]
    non_json_files = ["test.txt", "summary.pdf", "data.csv"]

    for filename in json_files + non_json_files:
        create_test_json_file(temp_folder, filename)

    with patch("src.qa_gpt.core.ui.materials_overview.initialize_controllers") as mock_init:
        mock_controller = MagicMock()
        mock_init.return_value = mock_controller
        mock_controller.get_material_table.return_value = {}
        mock_controller.get_material_by_filename.return_value = mock_file_meta_no_comments

        result = get_all_materials(temp_folder)

        assert len(result) == 2  # Only JSON files should be processed
        names = [material["question_set_name"] for material in result]
        assert "StandardSummary_bullet_points_0" in names
        assert "TechnicalSummary_overview_0" in names


def test_get_all_materials_complex_folder_structure(temp_folder, mock_file_meta_with_comments):
    """Test get_all_materials with complex nested folder structure."""
    # Create complex folder structure
    folders = ["Mamba_0", "Gemma_1", "CausalMMM: Learning Causal Structure for Marketing Mix_3"]

    for folder in folders:
        folder_path = os.path.join(temp_folder, folder)
        os.makedirs(folder_path, exist_ok=True)

        # Add different JSON files to each folder
        if folder == "Mamba_0":
            create_test_json_file(folder_path, "mc_question_StandardSummary_bullet_points_0.json")
        elif folder == "Gemma_1":
            create_test_json_file(folder_path, "mc_question_TechnicalSummary_overview_0.json")
        else:
            create_test_json_file(folder_path, "mc_question_InnovationSummary_key_concepts_0.json")

    with patch("src.qa_gpt.core.ui.materials_overview.initialize_controllers") as mock_init:
        mock_controller = MagicMock()
        mock_init.return_value = mock_controller
        mock_controller.get_material_table.return_value = {}
        mock_controller.get_material_by_filename.return_value = mock_file_meta_with_comments

        result = get_all_materials(temp_folder)

        assert len(result) == 3
        paths = [material["material_name"] for material in result]
        assert "Mamba" in paths  # Implementation splits on "_" and takes first part
        assert "Gemma" in paths  # Implementation splits on "_" and takes first part
        assert (
            "CausalMMM: Learning Causal Structure for Marketing Mix" in paths
        )  # Implementation splits on "_" and takes first part


def test_get_all_materials_custom_folder_path(temp_folder, mock_file_meta_no_comments):
    """Test get_all_materials with custom folder path."""
    filename = "mc_question_StandardSummary_bullet_points_0.json"
    create_test_json_file(temp_folder, filename)

    with patch("src.qa_gpt.core.ui.materials_overview.initialize_controllers") as mock_init:
        mock_controller = MagicMock()
        mock_init.return_value = mock_controller
        mock_controller.get_material_table.return_value = {}
        mock_controller.get_material_by_filename.return_value = mock_file_meta_no_comments

        result = get_all_materials(temp_folder)

        assert len(result) == 1
        material = result[0]
        assert material["question_set_name"] == "StandardSummary_bullet_points_0"
        assert material["full_path"] == os.path.join(temp_folder, filename)


def test_get_all_materials_question_set_id_extraction():
    """Test that question set ID is correctly extracted from various filename patterns."""
    test_cases = [
        ("mc_question_StandardSummary_bullet_points_0.json", "StandardSummary_bullet_points_0"),
        ("mc_question_TechnicalSummary_overview_0.json", "TechnicalSummary_overview_0"),
        ("mc_question_InnovationSummary_key_concepts_0.json", "InnovationSummary_key_concepts_0"),
        ("mc_question_0.json", "0"),
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        for filename, expected_id in test_cases:
            create_test_json_file(temp_dir, filename)

            with patch("src.qa_gpt.core.ui.materials_overview.initialize_controllers") as mock_init:
                mock_controller = MagicMock()
                mock_init.return_value = mock_controller
                mock_controller.get_material_table.return_value = {}
                mock_controller.get_material_by_filename.return_value = None

                result = get_all_materials(temp_dir)

                assert len(result) == 1
                assert result[0]["question_set_name"] == expected_id

                # Clean up for next iteration
                os.remove(os.path.join(temp_dir, filename))


def test_get_all_materials_controller_initialization_error(temp_folder):
    """Test get_all_materials when controller initialization fails."""
    with patch("src.qa_gpt.core.ui.materials_overview.initialize_controllers") as mock_init:
        mock_init.side_effect = Exception("Controller initialization failed")

        with pytest.raises(Exception, match="Controller initialization failed"):
            get_all_materials(temp_folder)


def test_get_all_materials_material_table_access_error(temp_folder):
    """Test get_all_materials when material table access fails."""
    filename = "mc_question_StandardSummary_bullet_points_0.json"
    create_test_json_file(temp_folder, filename)

    with patch("src.qa_gpt.core.ui.materials_overview.initialize_controllers") as mock_init:
        mock_controller = MagicMock()
        mock_init.return_value = mock_controller
        mock_controller.get_material_by_filename.side_effect = Exception("Database access failed")

        with pytest.raises(Exception, match="Database access failed"):
            get_all_materials(temp_folder)
