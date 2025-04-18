from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.qa_gpt.core.objects.questions import (
    Choice,
    MultipleChoiceQuestion,
    MultipleChoiceQuestionSet,
)
from src.qa_gpt.core.objects.summaries import (
    BulletPoint,
    Conclusion,
    InnovationSummary,
    Motivation,
    StandardSummary,
    TechnicalSummary,
)
from src.qa_gpt.core.utils.fetch_utils import (
    fetch_material_add_sets,
    fetch_material_add_summary,
    output_question_data,
)


@pytest.fixture
def test_pdf_folder(tmp_path):
    # Create a test PDF folder with a sample PDF
    pdf_folder = tmp_path / "pdf_data"
    pdf_folder.mkdir()
    (pdf_folder / "test.pdf").touch()
    return pdf_folder


@pytest.fixture
def mock_qa_controller():
    controller = MagicMock()
    with patch("src.qa_gpt.core.utils.fetch_utils.QAController") as mock:
        mock.return_value = controller
        yield controller


@pytest.fixture
def mock_material_controller():
    controller = MagicMock()
    with patch("src.qa_gpt.core.utils.fetch_utils.MaterialController") as mock:
        mock.return_value = controller
        yield controller


@pytest.fixture
def mock_db_controller():
    controller = MagicMock()
    with patch("src.qa_gpt.core.utils.fetch_utils.LocalDatabaseController") as mock:
        mock.return_value = controller
        yield controller


@pytest.mark.asyncio
async def test_fetch_material_add_summary_flow(
    test_pdf_folder, mock_qa_controller, mock_material_controller, mock_db_controller
):
    # Setup mock returns
    file_meta = MagicMock()
    file_meta.summaries = {
        "StandardSummary": MagicMock(),  # Already exists
        "TechnicalSummary": None,  # Will be added
        "InnovationSummary": None,  # Will be added
    }
    file_meta.__getitem__.return_value = str(test_pdf_folder / "test.pdf")

    def append_summary_side_effect(file_id, summary):
        summary_type = summary.__class__.__name__
        file_meta.summaries[summary_type] = summary

    mock_material_controller.get_material_table.return_value = {"test_id": file_meta}
    mock_material_controller.append_summary.side_effect = append_summary_side_effect

    # Create test summaries
    technical_summary = TechnicalSummary(
        overview="Test overview",
        key_concepts=["concept1", "concept2"],
        technical_details=["Detail 1: value1"],
        implementation_steps=["step1", "step2"],
        requirements=["Requirement 1: value1"],
        limitations=["limit1"],
    )

    innovation_summary = InnovationSummary(
        overview="Test overview",
        key_concepts=["concept1", "concept2"],
        innovation_points=[],
        references=["ref1", "ref2"],
    )

    # Setup mock to return coroutines for async functions
    async def mock_get_summaries_batch(*args, **kwargs):
        return [technical_summary, innovation_summary]

    mock_qa_controller.get_summaries_batch.side_effect = mock_get_summaries_batch

    # Run the function
    with patch("src.qa_gpt.core.utils.fetch_utils.Path") as mock_path:
        mock_path.return_value = test_pdf_folder
        await fetch_material_add_summary(process_all=True)

    # Verify function calls
    mock_material_controller.fetch_material_folder.assert_called_once_with(test_pdf_folder)


@pytest.mark.asyncio
async def test_fetch_material_add_sets_flow(
    test_pdf_folder, mock_qa_controller, mock_material_controller, mock_db_controller
):
    # Setup mock returns
    file_meta = MagicMock()
    file_meta.mc_question_sets = {
        "StandardSummary_motivation_0": MagicMock(),  # Already exists
        "StandardSummary_conclusion_0": MagicMock(),  # Already exists
        "StandardSummary_bullet_points_0": MagicMock(),  # Already exists
    }

    # Create mock summaries with model_dump
    standard_summary = MagicMock()
    standard_summary.model_dump.return_value = {
        "summary_type": "standard",
        "motivation": "test motivation",
        "conclusion": "test conclusion",
        "bullet_points": "test bullet points",
    }

    technical_summary = MagicMock()
    technical_summary.model_dump.return_value = {
        "summary_type": "technical",
        "overview": "test overview",
        "key_concepts": ["test concept"],
        "technical_details": ["test detail"],
        "implementation_steps": ["test step"],
        "requirements": ["test requirement"],
        "limitations": ["test limitation"],
    }

    file_meta.summaries = {
        "StandardSummary": standard_summary,
        "TechnicalSummary": technical_summary,
    }
    file_meta.__getitem__.return_value = str(test_pdf_folder / "test.pdf")

    def append_mc_question_set_side_effect(file_id, question_set, prefix=""):
        file_meta.mc_question_sets[prefix] = question_set

    mock_material_controller.get_material_table.return_value = {"test_id": file_meta}
    mock_material_controller.append_mc_question_set.side_effect = append_mc_question_set_side_effect

    # Create a test question set
    question = MultipleChoiceQuestion(
        question_description="Test question",
        choice_1=Choice(choice_description="Choice 1", answer=True, explanation="Explanation 1"),
        choice_2=Choice(choice_description="Choice 2", answer=False, explanation="Explanation 2"),
        choice_3=Choice(choice_description="Choice 3", answer=False, explanation="Explanation 3"),
        choice_4=Choice(choice_description="Choice 4", answer=False, explanation="Explanation 4"),
    )

    question_set = MultipleChoiceQuestionSet(
        question_1=question,
        question_2=question,
        question_3=question,
        question_4=question,
        question_5=question,
    )

    # Setup mock to return coroutines for async functions
    async def mock_get_questions_batch(*args, **kwargs):
        return [question_set] * len(args[0])  # Return a question set for each file path

    mock_qa_controller.get_questions_batch.side_effect = mock_get_questions_batch

    # Run the function
    with patch("src.qa_gpt.core.utils.fetch_utils.Path") as mock_path:
        mock_path.return_value = test_pdf_folder
        await fetch_material_add_sets(process_all=True)

    # Verify function calls
    mock_material_controller.fetch_material_folder.assert_called_once_with(test_pdf_folder)


def test_output_question_data_flow(
    test_pdf_folder, mock_qa_controller, mock_material_controller, mock_db_controller
):
    # Setup output folder
    output_folder = Path("./output_question_data")

    # Run the function
    output_question_data(process_all=True)

    # Verify the flow
    mock_material_controller.output_material_as_folder.assert_called_once_with(output_folder)


@pytest.mark.asyncio
async def test_full_script_flow(
    test_pdf_folder, mock_qa_controller, mock_material_controller, mock_db_controller
):
    # Setup mock returns
    file_meta = MagicMock()
    file_meta.summaries = {}
    file_meta.mc_question_sets = {}
    file_meta.__getitem__.return_value = str(test_pdf_folder / "test.pdf")

    def append_mc_question_set_side_effect(file_id, question_set, prefix=""):
        file_meta.mc_question_sets[prefix] = question_set

    def append_summary_side_effect(file_id, summary):
        summary_type = summary.__class__.__name__
        file_meta.summaries[summary_type] = summary

    mock_material_controller.get_material_table.return_value = {"test_id": file_meta}
    mock_material_controller.append_mc_question_set.side_effect = append_mc_question_set_side_effect
    mock_material_controller.append_summary.side_effect = append_summary_side_effect

    # Create test summaries
    standard_summary = StandardSummary(
        motivation=Motivation(
            description="Test description",
            problem_to_solve="Test problem",
            how_to_solve="Test solution",
            why_can_be_solved="Test reason",
        ),
        conclusion=Conclusion(
            description="Test description",
            problem_to_solve="Test problem",
            how_much_is_solved="Test progress",
            contribution="Test contribution",
        ),
        bullet_points=[
            BulletPoint(
                subject="Test subject",
                description="Test description",
                technical_details="Test details",
                importance_explanation="Test importance",
                importance=1,
            )
        ],
    )

    technical_summary = TechnicalSummary(
        overview="Test overview",
        key_concepts=["concept1", "concept2"],
        technical_details=["Detail 1: value1"],
        implementation_steps=["step1", "step2"],
        requirements=["Requirement 1: value1"],
        limitations=["limit1"],
    )

    innovation_summary = InnovationSummary(
        overview="Test overview",
        key_concepts=["concept1", "concept2"],
        innovation_points=[],
        references=["ref1", "ref2"],
    )

    # Create a test question set
    question = MultipleChoiceQuestion(
        question_description="Test question",
        choice_1=Choice(choice_description="Choice 1", answer=True, explanation="Explanation 1"),
        choice_2=Choice(choice_description="Choice 2", answer=False, explanation="Explanation 2"),
        choice_3=Choice(choice_description="Choice 3", answer=False, explanation="Explanation 3"),
        choice_4=Choice(choice_description="Choice 4", answer=False, explanation="Explanation 4"),
    )

    question_set = MultipleChoiceQuestionSet(
        question_1=question,
        question_2=question,
        question_3=question,
        question_4=question,
        question_5=question,
    )

    # Setup mocks to return coroutines for async functions
    async def mock_get_summaries_batch(*args, **kwargs):
        return [standard_summary, technical_summary, innovation_summary]

    async def mock_get_questions_batch(*args, **kwargs):
        return [question_set] * len(args[0])  # Return a question set for each file path

    mock_qa_controller.get_summaries_batch.side_effect = mock_get_summaries_batch
    mock_qa_controller.get_questions_batch.side_effect = mock_get_questions_batch

    # Run all functions in sequence
    with patch("src.qa_gpt.core.utils.fetch_utils.Path") as mock_path:
        mock_path.return_value = test_pdf_folder
        await fetch_material_add_summary(process_all=True)
        await fetch_material_add_sets(process_all=True)

    # Verify function calls
    assert mock_material_controller.fetch_material_folder.call_count == 2


@pytest.mark.asyncio
async def test_fetch_material_add_summary_single_file(
    test_pdf_folder, mock_qa_controller, mock_material_controller, mock_db_controller
):
    # Setup mock returns
    file_meta = MagicMock()
    file_meta.summaries = {
        "StandardSummary": MagicMock(),  # Already exists
        "TechnicalSummary": None,  # Will be added
        "InnovationSummary": None,  # Will be added
    }
    file_meta.__getitem__.return_value = str(test_pdf_folder / "test.pdf")

    def append_summary_side_effect(file_id, summary):
        summary_type = summary.__class__.__name__
        file_meta.summaries[summary_type] = summary

    mock_material_controller.get_material_table.return_value = {
        "test_id": file_meta,
        "other_id": MagicMock(),  # This should be ignored
    }
    mock_material_controller.append_summary.side_effect = append_summary_side_effect

    # Create test summaries
    technical_summary = TechnicalSummary(
        overview="Test overview",
        key_concepts=["concept1", "concept2"],
        technical_details=["Detail 1: value1"],
        implementation_steps=["step1", "step2"],
        requirements=["Requirement 1: value1"],
        limitations=["limit1"],
    )

    innovation_summary = InnovationSummary(
        overview="Test overview",
        key_concepts=["concept1", "concept2"],
        innovation_points=[],
        references=["ref1", "ref2"],
    )

    # Setup mock to return coroutines for async functions
    async def mock_get_summaries_batch(*args, **kwargs):
        return [technical_summary, innovation_summary]

    mock_qa_controller.get_summaries_batch.side_effect = mock_get_summaries_batch

    # Run the function with single file mode
    with patch("src.qa_gpt.core.utils.fetch_utils.Path") as mock_path:
        mock_path.return_value = test_pdf_folder
        await fetch_material_add_summary(file_id="test_id", process_all=True)

    # Verify function calls
    mock_material_controller.fetch_material_folder.assert_called_once_with(test_pdf_folder)
    # Verify that append_summary was only called for the specified file
    assert (
        mock_material_controller.append_summary.call_count == 2
    )  # For Technical and Innovation summaries


@pytest.mark.asyncio
async def test_fetch_material_add_sets_single_file(
    test_pdf_folder, mock_qa_controller, mock_material_controller, mock_db_controller
):
    # Setup mock returns
    file_meta = MagicMock()
    file_meta.mc_question_sets = {
        "StandardSummary_motivation_0": MagicMock(),  # Already exists
        "StandardSummary_conclusion_0": MagicMock(),  # Already exists
        "StandardSummary_bullet_points_0": MagicMock(),  # Already exists
    }

    # Create mock summaries with model_dump
    standard_summary = MagicMock()
    standard_summary.model_dump.return_value = {
        "summary_type": "standard",
        "motivation": "test motivation",
        "conclusion": "test conclusion",
        "bullet_points": "test bullet points",
    }

    technical_summary = MagicMock()
    technical_summary.model_dump.return_value = {
        "summary_type": "technical",
        "overview": "test overview",
        "key_concepts": ["test concept"],
        "technical_details": ["test detail"],
        "implementation_steps": ["test step"],
        "requirements": ["test requirement"],
        "limitations": ["test limitation"],
    }

    file_meta.summaries = {
        "StandardSummary": standard_summary,
        "TechnicalSummary": technical_summary,
    }
    file_meta.__getitem__.return_value = str(test_pdf_folder / "test.pdf")

    def append_mc_question_set_side_effect(file_id, question_set, prefix=""):
        file_meta.mc_question_sets[prefix] = question_set

    mock_material_controller.get_material_table.return_value = {
        "test_id": file_meta,
        "other_id": MagicMock(),  # This should be ignored
    }
    mock_material_controller.append_mc_question_set.side_effect = append_mc_question_set_side_effect

    # Create a test question set
    question = MultipleChoiceQuestion(
        question_description="Test question",
        choice_1=Choice(choice_description="Choice 1", answer=True, explanation="Explanation 1"),
        choice_2=Choice(choice_description="Choice 2", answer=False, explanation="Explanation 2"),
        choice_3=Choice(choice_description="Choice 3", answer=False, explanation="Explanation 3"),
        choice_4=Choice(choice_description="Choice 4", answer=False, explanation="Explanation 4"),
    )

    question_set = MultipleChoiceQuestionSet(
        question_1=question,
        question_2=question,
        question_3=question,
        question_4=question,
        question_5=question,
    )

    # Setup mock to return coroutines for async functions
    async def mock_get_questions_batch(*args, **kwargs):
        return [question_set] * len(args[0])  # Return a question set for each file path

    mock_qa_controller.get_questions_batch.side_effect = mock_get_questions_batch

    # Run the function with single file mode
    with patch("src.qa_gpt.core.utils.fetch_utils.Path") as mock_path:
        mock_path.return_value = test_pdf_folder
        await fetch_material_add_sets(file_id="test_id", process_all=True)

    # Verify function calls
    mock_material_controller.fetch_material_folder.assert_called_once_with(test_pdf_folder)
    # Verify that append_mc_question_set was only called for the specified file
    assert mock_material_controller.append_mc_question_set.call_count > 0


def test_output_question_data_single_file(
    test_pdf_folder, mock_qa_controller, mock_material_controller, mock_db_controller
):
    # Setup mock material table
    file_meta = MagicMock()
    mock_material_controller.get_material_table.return_value = {
        "test_id": file_meta,
        "other_id": MagicMock(),  # This should be ignored
    }

    # Run the function with single file mode
    output_question_data(file_id="test_id", process_all=True)

    # Verify the flow
    mock_material_controller.output_material_as_folder.assert_called_once_with(
        Path("./output_question_data")
    )


@pytest.mark.asyncio
async def test_fetch_material_add_summary_invalid_file_id(
    test_pdf_folder, mock_qa_controller, mock_material_controller, mock_db_controller
):
    # Setup mock returns
    mock_material_controller.get_material_table.return_value = {"test_id": MagicMock()}

    # Run the function with invalid file_id
    with pytest.raises(ValueError, match="File ID invalid_id not found in material table"):
        await fetch_material_add_summary(file_id="invalid_id", process_all=True)


@pytest.mark.asyncio
async def test_fetch_material_add_sets_invalid_file_id(
    test_pdf_folder, mock_qa_controller, mock_material_controller, mock_db_controller
):
    # Setup mock returns
    mock_material_controller.get_material_table.return_value = {"test_id": MagicMock()}

    # Run the function with invalid file_id
    with pytest.raises(ValueError, match="File ID invalid_id not found in material table"):
        await fetch_material_add_sets(file_id="invalid_id", process_all=True)


def test_output_question_data_invalid_file_id(
    test_pdf_folder, mock_qa_controller, mock_material_controller, mock_db_controller
):
    # Setup mock returns
    mock_material_controller.get_material_table.return_value = {"test_id": MagicMock()}

    # Run the function with invalid file_id
    with pytest.raises(ValueError, match="File ID invalid_id not found in material table"):
        output_question_data(file_id="invalid_id", process_all=True)
