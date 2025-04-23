import shutil
from pathlib import Path

import pytest
import streamlit as st

from src.qa_gpt.core.controller.db_controller import (
    LocalDatabaseController,
    MaterialController,
)
from src.qa_gpt.core.objects.materials import FileMeta
from src.qa_gpt.core.ui.material_operations import (
    display_material_operations,
    remove_material,
)


@pytest.fixture
def test_db_controller():
    db_name = "test_material_ops_db"
    controller = LocalDatabaseController(db_name=db_name)
    yield controller
    # Cleanup
    controller.db_path.unlink()


@pytest.fixture
def test_material_controller(test_db_controller):
    archive_name = "test_material_ops_archive"
    controller = MaterialController(db_controller=test_db_controller, archive_name=archive_name)
    yield controller
    # Cleanup
    shutil.rmtree(controller.archive_path, ignore_errors=True)


@pytest.fixture
def test_output_folder():
    folder_path = Path("test_output_question_data")
    folder_path.mkdir(exist_ok=True)
    yield folder_path
    # Cleanup
    shutil.rmtree(folder_path, ignore_errors=True)


@pytest.fixture
def test_pdf_folder():
    folder_path = Path("./pdf_data")
    folder_path.mkdir(exist_ok=True)
    yield folder_path
    # Cleanup
    shutil.rmtree(folder_path, ignore_errors=True)


@pytest.fixture
def test_material_folder(test_output_folder, test_material_controller, test_pdf_folder):
    # Create a test material in the database
    file_meta = FileMeta(
        id=0,
        file_name="test_material",
        file_suffix=".pdf",
        file_path=Path("test_material.pdf"),
        mc_question_sets={},
        summaries={},
        parsing_results={"sections": None, "images": None, "tables": None},
    )

    # Save to database
    test_material_controller.db_controller.save_data(
        file_meta,
        test_material_controller.db_controller.get_target_path(
            [test_material_controller.db_table_name, "0"]
        ),
    )

    # Save to mapping table
    test_material_controller.db_controller.save_data(
        "0",
        test_material_controller.db_controller.get_target_path(
            [test_material_controller.db_mapping_table_name, "test_material"]
        ),
    )

    # Create output folder structure
    material_folder = test_output_folder / "test_material_0"
    material_folder.mkdir(exist_ok=True)

    # Create some test files
    (material_folder / "meta_data.json").touch()
    (material_folder / "summary.json").touch()
    (material_folder / "mc_question_0.json").touch()

    # Create test PDF file
    pdf_path = test_pdf_folder / "test_material.pdf"
    pdf_path.touch()

    return material_folder


def test_remove_material_success(test_material_folder, test_material_controller):
    """Test successful material removal."""
    # Get the PDF file path
    pdf_path = Path("./pdf_data") / "test_material.pdf"

    # Verify material exists before removal
    assert test_material_folder.exists()
    assert pdf_path.exists()
    assert test_material_controller.get_material_table()
    assert test_material_controller.get_material_mapping_table()

    # Remove the material
    remove_material(str(test_material_folder), test_material_controller)

    # Verify material is removed from filesystem
    assert not test_material_folder.exists()
    assert not pdf_path.exists()  # Verify PDF is removed

    # Verify material is removed from database
    assert not test_material_controller.get_material_table()
    assert not test_material_controller.get_material_mapping_table()


def test_remove_material_not_found(test_output_folder, test_material_controller):
    """Test material removal when folder doesn't exist."""
    non_existent_folder = test_output_folder / "non_existent_material_0"

    # Remove the non-existent material
    remove_material(str(non_existent_folder), test_material_controller)

    # Verify folder still doesn't exist
    assert not non_existent_folder.exists()


def test_remove_material_database_error(
    test_material_folder, test_material_controller, monkeypatch
):
    """Test material removal when database operation fails."""
    # Get the PDF file path
    pdf_path = Path("./pdf_data") / "test_material.pdf"

    # Mock the remove_material_by_filename to return error
    def mock_remove(*args, **kwargs):
        return -1

    monkeypatch.setattr(test_material_controller, "remove_material_by_filename", mock_remove)

    # Verify material exists before removal attempt
    assert test_material_folder.exists()
    assert pdf_path.exists()

    # Try to remove the material
    remove_material(str(test_material_folder), test_material_controller)

    # Verify material still exists (removal should have failed)
    assert test_material_folder.exists()
    assert pdf_path.exists()  # Verify PDF still exists


def test_display_material_operations_no_material(test_material_controller):
    """Test display_material_operations when no material is selected."""
    # Mock streamlit functions
    st.header = lambda x: None
    st.write = lambda x: None

    # Test with no material selected
    display_material_operations(None, test_material_controller)

    # No assertions needed as we're just testing it doesn't raise exceptions


def test_display_material_operations_with_material(test_material_folder, test_material_controller):
    """Test display_material_operations when a material is selected."""
    # Mock streamlit functions
    st.header = lambda x: None
    st.selectbox = lambda *args, **kwargs: "Remove selected material"
    st.button = lambda *args, **kwargs: True
    st.error = lambda x: None
    st.success = lambda x: None
    st.session_state = {}
    st.rerun = lambda: None

    # Test with material selected
    display_material_operations(str(test_material_folder), test_material_controller)

    # No assertions needed as we're just testing it doesn't raise exceptions
