import json
import shutil
from pathlib import Path

import pytest

from src.qa_gpt.core.controller.db_controller import (
    LocalDatabaseController,
    MaterialController,
)
from src.qa_gpt.core.objects.materials import FileMeta
from src.qa_gpt.core.objects.questions import (
    Choice,
    MultipleChoiceQuestion,
    MultipleChoiceQuestionSet,
    QuestionComment,
)
from src.qa_gpt.core.objects.summaries import (
    BulletPoint,
    Conclusion,
    Motivation,
    StandardSummary,
    TechnicalSummary,
)


@pytest.fixture
def test_db_controller():
    db_name = "test_material_db"
    controller = LocalDatabaseController(db_name=db_name)
    yield controller
    # Cleanup
    controller.db_path.unlink()


@pytest.fixture
def test_material_controller(test_db_controller):
    archive_name = "test_material_archive"
    controller = MaterialController(db_controller=test_db_controller, archive_name=archive_name)
    yield controller
    # Cleanup
    shutil.rmtree(controller.archive_path, ignore_errors=True)


@pytest.fixture
def sample_choice():
    return Choice(choice_description="Test choice", answer=True, explanation="Test explanation")


@pytest.fixture
def sample_question(sample_choice):
    return MultipleChoiceQuestion(
        question_description="Test question",
        choice_1=sample_choice,
        choice_2=sample_choice,
        choice_3=sample_choice,
        choice_4=sample_choice,
    )


@pytest.fixture
def sample_question_set(sample_question):
    return MultipleChoiceQuestionSet(
        question_1=sample_question,
        question_2=sample_question,
        question_3=sample_question,
        question_4=sample_question,
        question_5=sample_question,
    )


@pytest.fixture
def sample_summary():
    motivation = Motivation(
        description="Test motivation",
        problem_to_solve="Test problem",
        how_to_solve="Test solution",
        why_can_be_solved="Test why",
    )
    conclusion = Conclusion(
        description="Test conclusion",
        problem_to_solve="Test problem",
        how_much_is_solved="Test solution",
        contribution="Test contribution",
    )
    bullet_point = BulletPoint(
        subject="Test subject",
        description="Test description",
        technical_details="Test details",
        importance_explanation="Test importance",
        importance=1,
    )
    return StandardSummary(
        motivation=motivation, conclusion=conclusion, bullet_points=[bullet_point]
    )


@pytest.fixture
def sample_technical_summary():
    return TechnicalSummary(
        overview="This technical document describes the implementation of a new machine learning algorithm",
        key_concepts=["Neural Networks", "Gradient Descent", "Backpropagation", "Loss Functions"],
        technical_details=[
            "Model architecture: Multi-layer perceptron with 3 hidden layers",
            "Activation function: ReLU for hidden layers, Softmax for output",
            "Optimization: Adam optimizer with learning rate 0.001",
            "Batch size: 32 samples per batch",
        ],
        implementation_steps=[
            "Data preprocessing and normalization",
            "Model architecture definition",
            "Training loop implementation",
            "Validation and testing procedures",
        ],
        requirements=[
            "Python: 3.8+",
            "TensorFlow: 2.4+",
            "CUDA: 11.0+",
            "RAM: 16GB minimum",
        ],
        limitations=[
            "High computational resource requirements",
            "Limited to supervised learning tasks",
            "Requires large labeled dataset",
        ],
    )


@pytest.fixture
def sample_question_comment():
    return QuestionComment(
        topic="difficulty",
        content="This question is too complex",
        is_positive=False,
        question_set_id="set_1",
        question_id="question_2",
    )


def test_material_controller_initialization(test_material_controller):
    assert test_material_controller.db_table_name == "material_table"
    assert test_material_controller.db_mapping_table_name == "material_id_mapping_table"
    assert test_material_controller.archive_path.exists()
    assert (
        test_material_controller.db_controller.get_data(test_material_controller.db_table_name)
        is not None
    )
    assert (
        test_material_controller.db_controller.get_data(
            test_material_controller.db_mapping_table_name
        )
        is not None
    )


def test_remove_dot_from_file_name():
    test_path = Path("test.file.name.pdf")
    new_path = MaterialController.remove_dot_from_file_name(test_path)
    assert new_path.name == "test_file_name.pdf"


def test_append_mc_question_set(test_material_controller, sample_question_set):
    # First create a file meta
    file_meta = FileMeta(
        id=0,
        file_name="test_file",
        file_suffix=".pdf",
        file_path=Path("test_file.pdf"),
        mc_question_sets={},
        question_comments={},
        summaries={},
        parsing_results={"sections": None, "images": None, "tables": None},
    )
    test_material_controller.db_controller.save_data(
        file_meta,
        test_material_controller.db_controller.get_target_path(
            [test_material_controller.db_table_name, "0"]
        ),
    )

    # Test appending question set without prefix
    result = test_material_controller.append_mc_question_set(0, sample_question_set)
    assert result == 0

    # Verify the question set was saved with numeric key
    updated_meta = test_material_controller.db_controller.get_data(
        test_material_controller.db_controller.get_target_path(
            [test_material_controller.db_table_name, "0"]
        )
    )
    assert "0" in updated_meta["mc_question_sets"]
    assert updated_meta["mc_question_sets"]["0"] == sample_question_set

    # Test appending multiple question sets with the same prefix
    prefix = "test_prefix"
    for _ in range(3):
        result = test_material_controller.append_mc_question_set(0, sample_question_set, prefix)
        assert result == 0

    # Verify the question sets were saved with incrementing IDs
    updated_meta = test_material_controller.db_controller.get_data(
        test_material_controller.db_controller.get_target_path(
            [test_material_controller.db_table_name, "0"]
        )
    )
    assert f"{prefix}_0" in updated_meta["mc_question_sets"]
    assert f"{prefix}_1" in updated_meta["mc_question_sets"]
    assert f"{prefix}_2" in updated_meta["mc_question_sets"]
    assert updated_meta["mc_question_sets"][f"{prefix}_0"] == sample_question_set
    assert updated_meta["mc_question_sets"][f"{prefix}_1"] == sample_question_set
    assert updated_meta["mc_question_sets"][f"{prefix}_2"] == sample_question_set


def test_append_summary(test_material_controller, sample_summary):
    # First create a file meta
    file_meta = FileMeta(
        id=0,
        file_name="test_file",
        file_suffix=".pdf",
        file_path=Path("test_file.pdf"),
        mc_question_sets={},
        question_comments={},
        summaries={},
        parsing_results={"sections": None, "images": None, "tables": None},
    )
    test_material_controller.db_controller.save_data(
        file_meta,
        test_material_controller.db_controller.get_target_path(
            [test_material_controller.db_table_name, "0"]
        ),
    )

    # Test appending summary
    result = test_material_controller.append_summary(0, sample_summary)
    assert result == 0

    # Verify the summary was saved
    updated_meta = test_material_controller.db_controller.get_data(
        test_material_controller.db_controller.get_target_path(
            [test_material_controller.db_table_name, "0"]
        )
    )
    assert "StandardSummary" in updated_meta["summaries"]
    assert updated_meta["summaries"]["StandardSummary"] == sample_summary
    assert isinstance(updated_meta["summaries"]["StandardSummary"], StandardSummary)
    assert updated_meta["summaries"]["StandardSummary"].motivation.description == "Test motivation"
    assert updated_meta["summaries"]["StandardSummary"].conclusion.description == "Test conclusion"
    assert len(updated_meta["summaries"]["StandardSummary"].bullet_points) == 1
    assert (
        updated_meta["summaries"]["StandardSummary"].bullet_points[0].description
        == "Test description"
    )


def test_append_technical_summary(test_material_controller, sample_technical_summary):
    # First create a file meta
    file_meta = FileMeta(
        id=0,
        file_name="test_file",
        file_suffix=".pdf",
        file_path=Path("test_file.pdf"),
        mc_question_sets={},
        question_comments={},
        summaries={},
        parsing_results={"sections": None, "images": None, "tables": None},
    )
    test_material_controller.db_controller.save_data(
        file_meta,
        test_material_controller.db_controller.get_target_path(
            [test_material_controller.db_table_name, "0"]
        ),
    )

    # Test appending technical summary
    result = test_material_controller.append_summary(0, sample_technical_summary)
    assert result == 0

    # Verify the technical summary was saved
    updated_meta = test_material_controller.db_controller.get_data(
        test_material_controller.db_controller.get_target_path(
            [test_material_controller.db_table_name, "0"]
        )
    )
    assert "TechnicalSummary" in updated_meta["summaries"]
    assert updated_meta["summaries"]["TechnicalSummary"] == sample_technical_summary
    assert isinstance(updated_meta["summaries"]["TechnicalSummary"], TechnicalSummary)
    assert (
        updated_meta["summaries"]["TechnicalSummary"].overview
        == "This technical document describes the implementation of a new machine learning algorithm"
    )
    assert "Neural Networks" in updated_meta["summaries"]["TechnicalSummary"].key_concepts
    assert any(
        "Model architecture:" in detail
        for detail in updated_meta["summaries"]["TechnicalSummary"].technical_details
    )


def test_append_multiple_summaries(
    test_material_controller, sample_summary, sample_technical_summary
):
    # First create a file meta
    file_meta = FileMeta(
        id=0,
        file_name="test_file",
        file_suffix=".pdf",
        file_path=Path("test_file.pdf"),
        mc_question_sets={},
        question_comments={},
        summaries={},
        parsing_results={"sections": None, "images": None, "tables": None},
    )
    test_material_controller.db_controller.save_data(
        file_meta,
        test_material_controller.db_controller.get_target_path(
            [test_material_controller.db_table_name, "0"]
        ),
    )

    # Test appending both summaries
    result1 = test_material_controller.append_summary(0, sample_summary)
    result2 = test_material_controller.append_summary(0, sample_technical_summary)
    assert result1 == 0
    assert result2 == 0

    # Verify both summaries were saved
    updated_meta = test_material_controller.db_controller.get_data(
        test_material_controller.db_controller.get_target_path(
            [test_material_controller.db_table_name, "0"]
        )
    )
    assert len(updated_meta["summaries"]) == 2
    assert "StandardSummary" in updated_meta["summaries"]
    assert "TechnicalSummary" in updated_meta["summaries"]
    assert isinstance(updated_meta["summaries"]["StandardSummary"], StandardSummary)
    assert isinstance(updated_meta["summaries"]["TechnicalSummary"], TechnicalSummary)


def test_get_material_table(test_material_controller):
    # Create some test data
    file_meta = FileMeta(
        id=0,
        file_name="test_file",
        file_suffix=".pdf",
        file_path=Path("test_file.pdf"),
        mc_question_sets={},
        question_comments={},
        summaries={},
        parsing_results={"sections": None, "images": None, "tables": None},
    )
    test_material_controller.db_controller.save_data(
        file_meta,
        test_material_controller.db_controller.get_target_path(
            [test_material_controller.db_table_name, "0"]
        ),
    )

    # Test getting material table
    material_table = test_material_controller.get_material_table()
    assert "0" in material_table
    assert material_table["0"] == file_meta


def test_get_material_mapping_table(test_material_controller):
    # Create some test data
    file_meta = FileMeta(
        id=0,
        file_name="test_file",
        file_suffix=".pdf",
        file_path=Path("test_file.pdf"),
        mc_question_sets={},
        question_comments={},
        summaries={},
        parsing_results={"sections": None, "images": None, "tables": None},
    )
    test_material_controller.db_controller.save_data(
        file_meta,
        test_material_controller.db_controller.get_target_path(
            [test_material_controller.db_table_name, "0"]
        ),
    )
    test_material_controller.db_controller.save_data(
        "0",  # String ID
        test_material_controller.db_controller.get_target_path(
            [test_material_controller.db_mapping_table_name, "test_file"]
        ),
    )

    mapping_table = test_material_controller.get_material_mapping_table()
    assert isinstance(mapping_table, dict)
    assert "test_file" in mapping_table
    assert isinstance(mapping_table["test_file"], str)
    assert mapping_table["test_file"] == "0"


def test_mapping_table_string_ids(test_material_controller):
    # Create multiple test files
    for i in range(3):
        file_meta = FileMeta(
            id=i,
            file_name=f"test_file_{i}",
            file_suffix=".pdf",
            file_path=Path(f"test_file_{i}.pdf"),
            mc_question_sets={},
            question_comments={},
            summaries={},
            parsing_results={"sections": None, "images": None, "tables": None},
        )
        test_material_controller.db_controller.save_data(
            file_meta,
            test_material_controller.db_controller.get_target_path(
                [test_material_controller.db_table_name, str(i)]
            ),
        )
        test_material_controller.db_controller.save_data(
            str(i),  # String ID
            test_material_controller.db_controller.get_target_path(
                [test_material_controller.db_mapping_table_name, f"test_file_{i}"]
            ),
        )

    mapping_table = test_material_controller.get_material_mapping_table()
    assert len(mapping_table) == 3

    # Verify all IDs are strings
    for file_name, material_id in mapping_table.items():
        assert isinstance(material_id, str), f"Material ID for {file_name} should be a string"
        assert material_id.isdigit(), f"Material ID {material_id} should be a numeric string"


def test_output_material_as_folder(test_material_controller, sample_question_set, sample_summary):
    # Create test data
    file_meta = FileMeta(
        id=0,
        file_name="test_file",
        file_suffix=".pdf",
        file_path=Path("test_file.pdf"),
        mc_question_sets={"0": sample_question_set},
        question_comments={},
        summaries={"StandardSummary": sample_summary},
        parsing_results={"sections": None, "images": None, "tables": None},
    )
    test_material_controller.db_controller.save_data(
        file_meta,
        test_material_controller.db_controller.get_target_path(
            [test_material_controller.db_table_name, "0"]
        ),
    )

    # Create output directory
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)

    # Test output
    result = test_material_controller.output_material_as_folder(output_dir)
    assert result == 0

    # Verify output files
    material_dir = output_dir / "test_file"
    assert material_dir.exists()
    assert (material_dir / "meta_data.json").exists()
    assert (material_dir / "summary_StandardSummary.json").exists()
    assert (material_dir / "mc_question_0.json").exists()

    # Verify summary.json contains standard summary data
    with open(material_dir / "summary_StandardSummary.json") as f:
        summary_data = json.load(f)
        assert "summary_type" in summary_data
        assert summary_data["summary_type"] == "standard"
        assert "motivation" in summary_data
        assert "conclusion" in summary_data
        assert "bullet_points" in summary_data
        assert len(summary_data["bullet_points"]) == 1
        assert summary_data["motivation"]["description"] == "Test motivation"
        assert summary_data["conclusion"]["description"] == "Test conclusion"

    # Cleanup
    shutil.rmtree(output_dir)


def test_output_material_with_technical_summary(test_material_controller, sample_technical_summary):
    # Create test data with technical summary
    file_meta = FileMeta(
        id=0,
        file_name="test_file",
        file_suffix=".pdf",
        file_path=Path("test_file.pdf"),
        mc_question_sets={},
        question_comments={},
        summaries={"TechnicalSummary": sample_technical_summary},
        parsing_results={"sections": None, "images": None, "tables": None},
    )
    test_material_controller.db_controller.save_data(
        file_meta,
        test_material_controller.db_controller.get_target_path(
            [test_material_controller.db_table_name, "0"]
        ),
    )

    # Create output directory
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)

    # Test output
    result = test_material_controller.output_material_as_folder(output_dir)
    assert result == 0

    # Verify output files
    material_dir = output_dir / "test_file"
    assert material_dir.exists()
    assert (material_dir / "meta_data.json").exists()
    assert (material_dir / "summary_TechnicalSummary.json").exists()

    # Verify summary.json contains technical summary data
    with open(material_dir / "summary_TechnicalSummary.json") as f:
        summary_data = json.load(f)
        assert "overview" in summary_data
        assert "key_concepts" in summary_data
        assert "technical_details" in summary_data
        assert "implementation_steps" in summary_data
        assert "requirements" in summary_data
        assert "limitations" in summary_data
        assert (
            summary_data["overview"]
            == "This technical document describes the implementation of a new machine learning algorithm"
        )
        assert "Neural Networks" in summary_data["key_concepts"]
        assert any("Model architecture:" in detail for detail in summary_data["technical_details"])

    # Cleanup
    shutil.rmtree(output_dir)


def test_remove_material_by_filename(test_material_controller):
    # Create test data
    file_name = "test_file"
    file_meta = FileMeta(
        id=0,
        file_name=file_name,
        file_suffix=".pdf",
        file_path=test_material_controller.archive_path / f"{file_name}_0.pdf",
        mc_question_sets={},
        question_comments={},
        summaries={},
        parsing_results={"sections": None, "images": None, "tables": None},
    )

    # Save file meta to material table
    test_material_controller.db_controller.save_data(
        file_meta,
        test_material_controller.db_controller.get_target_path(
            [test_material_controller.db_table_name, "0"]
        ),
    )

    # Save mapping with string ID
    test_material_controller.db_controller.save_data(
        "0",  # String ID
        test_material_controller.db_controller.get_target_path(
            [test_material_controller.db_mapping_table_name, file_name]
        ),
    )

    # Create a dummy file
    file_meta["file_path"].touch()

    # Test removing the material
    result = test_material_controller.remove_material_by_filename(file_name)
    assert result == 0

    # Verify the file is deleted
    assert not file_meta["file_path"].exists()

    # Verify material table entry is deleted
    material_table = test_material_controller.get_material_table()
    assert "0" not in material_table

    # Verify mapping table entry is deleted
    mapping_table = test_material_controller.get_material_mapping_table()
    assert file_name not in mapping_table


def test_remove_material_by_filename_not_found(test_material_controller):
    # Test removing non-existent material
    result = test_material_controller.remove_material_by_filename("nonexistent_file")
    assert result == -1

    # Verify no changes were made to the tables
    material_table = test_material_controller.get_material_table()
    mapping_table = test_material_controller.get_material_mapping_table()
    assert material_table == {}
    assert mapping_table == {}


def test_remove_material_by_filename_with_associated_data(
    test_material_controller, sample_question_set, sample_summary
):
    # Create test data with associated question set and summary
    file_name = "test_file"
    file_meta = FileMeta(
        id=0,
        file_name=file_name,
        file_suffix=".pdf",
        file_path=test_material_controller.archive_path / f"{file_name}_0.pdf",
        mc_question_sets={"0": sample_question_set},
        question_comments={},
        summaries={"StandardSummary": sample_summary},
        parsing_results={"sections": None, "images": None, "tables": None},
    )

    # Save file meta to material table
    test_material_controller.db_controller.save_data(
        file_meta,
        test_material_controller.db_controller.get_target_path(
            [test_material_controller.db_table_name, "0"]
        ),
    )

    # Save mapping with string ID
    test_material_controller.db_controller.save_data(
        "0",  # String ID
        test_material_controller.db_controller.get_target_path(
            [test_material_controller.db_mapping_table_name, file_name]
        ),
    )

    # Create a dummy file
    file_meta["file_path"].touch()

    # Test removing the material
    result = test_material_controller.remove_material_by_filename(file_name)
    assert result == 0

    # Verify the file is deleted
    assert not file_meta["file_path"].exists()

    # Verify material table entry is deleted
    material_table = test_material_controller.get_material_table()
    assert "0" not in material_table

    # Verify mapping table entry is deleted
    mapping_table = test_material_controller.get_material_mapping_table()
    assert file_name not in mapping_table


def test_remove_material_with_technical_summary(test_material_controller, sample_technical_summary):
    # Create test data with technical summary
    file_name = "test_file"
    file_meta = FileMeta(
        id=0,
        file_name=file_name,
        file_suffix=".pdf",
        file_path=test_material_controller.archive_path / f"{file_name}_0.pdf",
        mc_question_sets={},
        question_comments={},
        summaries={"TechnicalSummary": sample_technical_summary},
        parsing_results={"sections": None, "images": None, "tables": None},
    )

    # Save file meta to material table
    test_material_controller.db_controller.save_data(
        file_meta,
        test_material_controller.db_controller.get_target_path(
            [test_material_controller.db_table_name, "0"]
        ),
    )

    # Save mapping with string ID
    test_material_controller.db_controller.save_data(
        "0",  # String ID
        test_material_controller.db_controller.get_target_path(
            [test_material_controller.db_mapping_table_name, file_name]
        ),
    )

    # Create a dummy file
    file_meta["file_path"].touch()

    # Test removing the material
    result = test_material_controller.remove_material_by_filename(file_name)
    assert result == 0

    # Verify the file is deleted
    assert not file_meta["file_path"].exists()

    # Verify material table entry is deleted
    material_table = test_material_controller.get_material_table()
    assert "0" not in material_table

    # Verify mapping table entry is deleted
    mapping_table = test_material_controller.get_material_mapping_table()
    assert file_name not in mapping_table


def test_append_question_comment(
    test_material_controller, sample_question_set, sample_question_comment
):
    # First create a file meta with a question set
    file_meta = FileMeta(
        id=0,
        file_name="test_file",
        file_suffix=".pdf",
        file_path=Path("test_file.pdf"),
        mc_question_sets={"set_1": sample_question_set},
        question_comments={},
        summaries={},
        parsing_results={"sections": None, "images": None, "tables": None},
    )
    test_material_controller.db_controller.save_data(
        file_meta,
        test_material_controller.db_controller.get_target_path(
            [test_material_controller.db_table_name, "0"]
        ),
    )

    # Test appending comment
    result = test_material_controller.append_question_comment(0, sample_question_comment)
    assert result == 0

    # Verify the comment was saved with correct key
    updated_meta = test_material_controller.db_controller.get_data(
        test_material_controller.db_controller.get_target_path(
            [test_material_controller.db_table_name, "0"]
        )
    )
    assert "set_1_question_2_0" in updated_meta["question_comments"]
    assert updated_meta["question_comments"]["set_1_question_2_0"] == sample_question_comment


def test_append_multiple_question_comments(
    test_material_controller, sample_question_set, sample_question_comment
):
    # First create a file meta with a question set
    file_meta = FileMeta(
        id=0,
        file_name="test_file",
        file_suffix=".pdf",
        file_path=Path("test_file.pdf"),
        mc_question_sets={"set_1": sample_question_set},
        question_comments={},
        summaries={},
        parsing_results={"sections": None, "images": None, "tables": None},
    )
    test_material_controller.db_controller.save_data(
        file_meta,
        test_material_controller.db_controller.get_target_path(
            [test_material_controller.db_table_name, "0"]
        ),
    )

    # Add multiple comments for the same question
    for i in range(3):
        comment = QuestionComment(
            topic=f"comment_{i}",
            content=f"Content {i}",
            is_positive=True,
            question_set_id="set_1",
            question_id="question_2",
        )
        result = test_material_controller.append_question_comment(0, comment)
        assert result == 0

    # Verify all comments were saved with correct keys
    updated_meta = test_material_controller.db_controller.get_data(
        test_material_controller.db_controller.get_target_path(
            [test_material_controller.db_table_name, "0"]
        )
    )
    assert "set_1_question_2_0" in updated_meta["question_comments"]
    assert "set_1_question_2_1" in updated_meta["question_comments"]
    assert "set_1_question_2_2" in updated_meta["question_comments"]


def test_append_question_comment_invalid_question_set(
    test_material_controller, sample_question_comment
):
    # Create file meta without the question set
    file_meta = FileMeta(
        id=0,
        file_name="test_file",
        file_suffix=".pdf",
        file_path=Path("test_file.pdf"),
        mc_question_sets={},
        question_comments={},
        summaries={},
        parsing_results={"sections": None, "images": None, "tables": None},
    )
    test_material_controller.db_controller.save_data(
        file_meta,
        test_material_controller.db_controller.get_target_path(
            [test_material_controller.db_table_name, "0"]
        ),
    )

    # Test appending comment to non-existent question set
    with pytest.raises(ValueError, match="Question set 'set_1' not found"):
        test_material_controller.append_question_comment(0, sample_question_comment)


def test_append_question_comment_invalid_question_id(test_material_controller, sample_question_set):
    # Create file meta with question set
    file_meta = FileMeta(
        id=0,
        file_name="test_file",
        file_suffix=".pdf",
        file_path=Path("test_file.pdf"),
        mc_question_sets={"set_1": sample_question_set},
        question_comments={},
        summaries={},
        parsing_results={"sections": None, "images": None, "tables": None},
    )
    test_material_controller.db_controller.save_data(
        file_meta,
        test_material_controller.db_controller.get_target_path(
            [test_material_controller.db_table_name, "0"]
        ),
    )

    # Test appending comment with invalid question ID
    invalid_comment = QuestionComment(
        topic="test",
        content="test content",
        is_positive=True,
        question_set_id="set_1",
        question_id="question_6",  # Invalid question ID
    )
    with pytest.raises(
        ValueError,
        match="Invalid question ID 'question_6'. Must be 'question_1' through 'question_5'",
    ):
        test_material_controller.append_question_comment(0, invalid_comment)
