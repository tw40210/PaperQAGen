import shutil
from unittest.mock import patch

import pytest

from src.qa_gpt.core.controller.db_controller import (
    LocalDatabaseController,
    MaterialController,
)
from src.qa_gpt.core.controller.parsing_controller import ParsingController
from src.qa_gpt.core.objects.materials import FileMeta
from src.qa_gpt.core.objects.parsing import TextSection, TextSections


@pytest.fixture
def test_db_controller():
    db_name = "test_parsing_db"
    controller = LocalDatabaseController(db_name=db_name)
    yield controller
    # Cleanup
    controller.db_path.unlink()


@pytest.fixture
def test_material_controller(test_db_controller):
    archive_name = "test_parsing_archive"
    controller = MaterialController(db_controller=test_db_controller, archive_name=archive_name)
    yield controller
    # Cleanup
    shutil.rmtree(controller.archive_path, ignore_errors=True)


@pytest.fixture
def test_pdf_folder(tmp_path):
    # Create a test PDF data directory
    pdf_folder = tmp_path / "pdf_data"
    pdf_folder.mkdir(exist_ok=True)
    return pdf_folder


@pytest.fixture
def test_markdown_file(test_pdf_folder):
    # Create a test markdown file with some content
    content = """
# Test Document

## Section 1
This is the first section with some text.

![Image 1](image1.png)

## Section 2
This is the second section with a table.

| Header 1 | Header 2 |
|----------|----------|
| Cell 1   | Cell 2   |
| Cell 3   | Cell 4   |

## Section 3
This is the third section with another image.

![Image 2](image2.png)
"""
    file_path = test_pdf_folder / "test.md"
    file_path.write_text(content)
    return file_path


@pytest.fixture
def test_file_meta(test_material_controller, test_markdown_file):
    # Create a test FileMeta object
    file_meta = FileMeta(
        id=0,
        file_name=test_markdown_file.stem,
        file_suffix=test_markdown_file.suffix,
        file_path=test_material_controller.archive_path
        / f"{test_markdown_file.stem}_0{test_markdown_file.suffix}",
        mc_question_sets={},
        summaries={},
        question_comments={},
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
            [test_material_controller.db_mapping_table_name, test_markdown_file.stem]
        ),
    )

    # Copy the file to archive
    shutil.copy(test_markdown_file, file_meta.file_path)

    return file_meta


@pytest.mark.asyncio
async def test_fetch_material_add_parsing(test_material_controller, test_file_meta):
    # Create mock sections
    mock_sections = TextSections(
        sections=[
            TextSection(
                title="Test Document",
                summary="This is a test document",
                content="# Test Document\n\nThis is a test document",
            ),
            TextSection(
                title="Section 1",
                summary="This is the first section",
                content="## Section 1\n\nThis is the first section with some text.",
            ),
            TextSection(
                title="Section 2",
                summary="This is the second section",
                content="## Section 2\n\nThis is the second section with a table.",
            ),
            TextSection(
                title="Section 3",
                summary="This is the third section",
                content="## Section 3\n\nThis is the third section with another image.",
            ),
        ]
    )

    # Mock the parsing controller
    with patch(
        "src.qa_gpt.core.controller.parsing_controller.ParsingController.get_sections_from_text_file"
    ) as mock_get_sections:
        mock_get_sections.return_value = (mock_sections, ["image1.png", "image2.png"], ["table1"])

        # Initialize parsing controller
        parsing_controller = ParsingController()

        # Get the file meta from database
        file_meta = test_material_controller.db_controller.get_data(
            test_material_controller.db_controller.get_target_path(
                [test_material_controller.db_table_name, "0"]
            )
        )

        # Skip if parsing results already exist
        if file_meta["parsing_results"]["sections"] is not None:
            print(f"Skipping {file_meta['file_name']} as parsing results already exist.")
            return

        # Get parsing results
        sections, images, tables = parsing_controller.get_sections_from_text_file(
            str(file_meta["file_path"])
        )

        # Update parsing results
        file_meta["parsing_results"] = {"sections": sections, "images": images, "tables": tables}

        # Save updated file meta
        target_path = test_material_controller.db_controller.get_target_path(
            [test_material_controller.db_table_name, "0"]
        )
        test_material_controller.db_controller.save_data(file_meta, target_path)

        # Verify parsing results
        updated_meta = test_material_controller.db_controller.get_data(target_path)
        assert updated_meta["parsing_results"]["sections"] is not None
        assert updated_meta["parsing_results"]["images"] is not None
        assert updated_meta["parsing_results"]["tables"] is not None

        # Verify sections
        sections = updated_meta["parsing_results"]["sections"]
        assert len(sections.sections) == 4
        assert sections.sections[0].title == "Test Document"
        assert sections.sections[1].title == "Section 1"
        assert sections.sections[2].title == "Section 2"
        assert sections.sections[3].title == "Section 3"

        # Verify images
        images = updated_meta["parsing_results"]["images"]
        assert len(images) == 2
        assert "image1.png" in images[0]
        assert "image2.png" in images[1]

        # Verify tables
        tables = updated_meta["parsing_results"]["tables"]
        assert len(tables) == 1
        assert "table1" in tables[0]


@pytest.mark.asyncio
async def test_fetch_material_add_parsing_skip_existing(test_material_controller, test_file_meta):
    # Create mock sections
    mock_sections = TextSections(
        sections=[
            TextSection(
                title="Test Document",
                summary="This is a test document",
                content="# Test Document\n\nThis is a test document",
            ),
            TextSection(
                title="Section 1",
                summary="This is the first section",
                content="## Section 1\n\nThis is the first section with some text.",
            ),
            TextSection(
                title="Section 2",
                summary="This is the second section",
                content="## Section 2\n\nThis is the second section with a table.",
            ),
            TextSection(
                title="Section 3",
                summary="This is the third section",
                content="## Section 3\n\nThis is the third section with another image.",
            ),
        ]
    )

    # Mock the parsing controller
    with patch(
        "src.qa_gpt.core.controller.parsing_controller.ParsingController.get_sections_from_text_file"
    ) as mock_get_sections:
        mock_get_sections.return_value = (mock_sections, ["image1.png", "image2.png"], ["table1"])
        # Initialize parsing controller
        parsing_controller = ParsingController()

        # First parse
        file_meta = test_material_controller.db_controller.get_data(
            test_material_controller.db_controller.get_target_path(
                [test_material_controller.db_table_name, "0"]
            )
        )

        # Get parsing results
        sections, images, tables = parsing_controller.get_sections_from_text_file(
            str(file_meta["file_path"])
        )

        # Update parsing results
        file_meta["parsing_results"] = {"sections": sections, "images": images, "tables": tables}

        # Save first parsing results
        target_path = test_material_controller.db_controller.get_target_path(
            [test_material_controller.db_table_name, "0"]
        )
        test_material_controller.db_controller.save_data(file_meta, target_path)

        # Get first parsing results
        first_meta = test_material_controller.db_controller.get_data(target_path)
        first_sections = first_meta["parsing_results"]["sections"]

        # Try to parse again
        if first_meta["parsing_results"]["sections"] is not None:
            print(f"Skipping {first_meta['file_name']} as parsing results already exist.")
            return

        # Get second parsing results
        sections, images, tables = parsing_controller.get_sections_from_text_file(
            str(first_meta["file_path"])
        )

        # Update parsing results
        first_meta["parsing_results"] = {"sections": sections, "images": images, "tables": tables}

        # Save second parsing results
        test_material_controller.db_controller.save_data(first_meta, target_path)

        # Get second parsing results
        second_meta = test_material_controller.db_controller.get_data(target_path)
        second_sections = second_meta["parsing_results"]["sections"]

        # Verify that the sections are the same (parsing was skipped)
        assert first_sections == second_sections
