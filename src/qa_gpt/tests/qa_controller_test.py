import asyncio
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.qa_gpt.core.controller.qa_controller import QAController
from src.qa_gpt.core.objects.questions import (
    Choice,
    MultipleChoiceQuestion,
    MultipleChoiceQuestionSet,
)
from src.qa_gpt.core.objects.summaries import (
    BulletPoint,
    Conclusion,
    Motivation,
    StandardSummary,
)


@pytest.fixture
def qa_controller():
    controller = QAController()
    # Mock the _pdf_to_text method to avoid file operations
    controller.preprocess_controller._pdf_to_text = MagicMock(return_value="Test PDF content")
    return controller


@pytest.fixture
def test_file_path():
    return Path("test.pdf")


@pytest.fixture
def mock_summary():
    return StandardSummary(
        motivation=Motivation(
            description="Test description",
            problem_to_solve="Test problem",
            how_to_solve="Test solution",
            why_can_be_solved="Test explanation",
        ),
        conclusion=Conclusion(
            description="Test conclusion",
            problem_to_solve="Test problem",
            how_much_is_solved="Test result",
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


@pytest.fixture
def mock_questions():
    return MultipleChoiceQuestionSet(
        question_1=MultipleChoiceQuestion(
            question_description="Test question 1",
            choice_1=Choice(
                choice_description="Choice 1", answer=True, explanation="Explanation 1"
            ),
            choice_2=Choice(
                choice_description="Choice 2", answer=False, explanation="Explanation 2"
            ),
            choice_3=Choice(
                choice_description="Choice 3", answer=False, explanation="Explanation 3"
            ),
            choice_4=Choice(
                choice_description="Choice 4", answer=False, explanation="Explanation 4"
            ),
        ),
        question_2=MultipleChoiceQuestion(
            question_description="Test question 2",
            choice_1=Choice(
                choice_description="Choice 1", answer=False, explanation="Explanation 1"
            ),
            choice_2=Choice(
                choice_description="Choice 2", answer=True, explanation="Explanation 2"
            ),
            choice_3=Choice(
                choice_description="Choice 3", answer=False, explanation="Explanation 3"
            ),
            choice_4=Choice(
                choice_description="Choice 4", answer=False, explanation="Explanation 4"
            ),
        ),
        question_3=MultipleChoiceQuestion(
            question_description="Test question 3",
            choice_1=Choice(
                choice_description="Choice 1", answer=False, explanation="Explanation 1"
            ),
            choice_2=Choice(
                choice_description="Choice 2", answer=False, explanation="Explanation 2"
            ),
            choice_3=Choice(
                choice_description="Choice 3", answer=True, explanation="Explanation 3"
            ),
            choice_4=Choice(
                choice_description="Choice 4", answer=False, explanation="Explanation 4"
            ),
        ),
        question_4=MultipleChoiceQuestion(
            question_description="Test question 4",
            choice_1=Choice(
                choice_description="Choice 1", answer=False, explanation="Explanation 1"
            ),
            choice_2=Choice(
                choice_description="Choice 2", answer=False, explanation="Explanation 2"
            ),
            choice_3=Choice(
                choice_description="Choice 3", answer=False, explanation="Explanation 3"
            ),
            choice_4=Choice(
                choice_description="Choice 4", answer=True, explanation="Explanation 4"
            ),
        ),
        question_5=MultipleChoiceQuestion(
            question_description="Test question 5",
            choice_1=Choice(
                choice_description="Choice 1", answer=False, explanation="Explanation 1"
            ),
            choice_2=Choice(
                choice_description="Choice 2", answer=False, explanation="Explanation 2"
            ),
            choice_3=Choice(
                choice_description="Choice 3", answer=False, explanation="Explanation 3"
            ),
            choice_4=Choice(
                choice_description="Choice 4", answer=True, explanation="Explanation 4"
            ),
        ),
    )


def test_get_summary(qa_controller, test_file_path, mock_summary, mocker):
    # Setup mock
    mock_get_response = mocker.patch(
        "src.qa_gpt.core.controller.qa_controller.get_chat_gpt_response_structure_async"
    )
    mock_get_response.return_value = mock_summary

    # Test
    result = asyncio.run(qa_controller.get_summary(test_file_path, StandardSummary))

    # Assertions
    assert isinstance(result, StandardSummary)
    assert result == mock_summary
    mock_get_response.assert_called_once()
    qa_controller.preprocess_controller._pdf_to_text.assert_called_once_with(test_file_path)


def test_get_questions(qa_controller, test_file_path, mock_questions, mocker):
    # Setup mock
    mock_get_response = mocker.patch(
        "src.qa_gpt.core.controller.qa_controller.get_chat_gpt_response_structure_async"
    )
    mock_get_response.return_value = mock_questions

    # Test
    result = asyncio.run(qa_controller.get_questions(test_file_path, "test_field", "test_value"))

    # Assertions
    assert isinstance(result, MultipleChoiceQuestionSet)
    assert result == mock_questions
    assert mock_get_response.call_count == 2
    qa_controller.preprocess_controller._pdf_to_text.assert_called_once_with(test_file_path)


def test_preprocess_controller_initialization(qa_controller):
    """Test that the preprocess controller is properly initialized"""
    assert qa_controller.preprocess_controller is not None


def test_message_templates_initialization(qa_controller):
    """Test that message templates are properly initialized"""
    # Check summary message template
    assert "role" in qa_controller.summary_message_temp
    assert "content" in qa_controller.summary_message_temp

    # Check question message template
    assert "role" in qa_controller.question_message_temp
    assert "content" in qa_controller.question_message_temp

    # Check user input template
    assert "role" in qa_controller.user_input_temp
    assert "content" in qa_controller.user_input_temp


def test_preprocess_integration(qa_controller, test_file_path, mocker):
    """Test that preprocessing is properly integrated with the controller"""
    # Setup
    mock_preprocess = mocker.patch(
        "src.qa_gpt.core.controller.qa_controller.PreprocessController.preprocess"
    )
    mock_preprocess.return_value = "Test content"

    # Test
    asyncio.run(qa_controller.get_summary(test_file_path, StandardSummary))

    # Assertions
    mock_preprocess.assert_called_once_with(test_file_path)
    assert mock_preprocess.call_count == 1


@pytest.mark.asyncio
async def test_get_summary_async(qa_controller, test_file_path, mock_summary, mocker):
    # Setup mock
    mock_get_response = mocker.patch(
        "src.qa_gpt.core.controller.qa_controller.get_chat_gpt_response_structure_async"
    )
    mock_get_response.return_value = mock_summary

    # Test
    result = await qa_controller.get_summary(test_file_path, StandardSummary)

    # Assertions
    assert isinstance(result, StandardSummary)
    assert result == mock_summary
    mock_get_response.assert_called_once()
    qa_controller.preprocess_controller._pdf_to_text.assert_called_once_with(test_file_path)


@pytest.mark.asyncio
async def test_get_questions_async(qa_controller, test_file_path, mock_questions, mocker):
    # Setup mock
    mock_get_response = mocker.patch(
        "src.qa_gpt.core.controller.qa_controller.get_chat_gpt_response_structure_async"
    )
    mock_get_response.return_value = mock_questions

    # Test
    result = await qa_controller.get_questions(test_file_path, "test_field", "test_value")

    # Assertions
    assert isinstance(result, MultipleChoiceQuestionSet)
    assert result == mock_questions
    assert mock_get_response.call_count == 2
    qa_controller.preprocess_controller._pdf_to_text.assert_called_once_with(test_file_path)


@pytest.mark.asyncio
async def test_get_summaries_batch_rate_limiting(
    qa_controller, test_file_path, mock_summary, mocker
):
    # Setup mock
    mock_get_response = mocker.patch(
        "src.qa_gpt.core.controller.qa_controller.get_chat_gpt_response_structure_async"
    )
    mock_get_response.return_value = mock_summary

    # Test with multiple files and summary classes
    file_paths = [test_file_path, test_file_path, test_file_path]
    summary_classes = [StandardSummary, StandardSummary, StandardSummary]

    start_time = time.time()
    results = await qa_controller.get_summaries_batch(file_paths, summary_classes)
    end_time = time.time()

    # Assertions
    assert len(results) == 3
    assert all(isinstance(result, StandardSummary) for result in results)
    assert mock_get_response.call_count == 3
    # Check that the total time is at least 2 seconds (3 calls with 1 second delay between each)
    assert end_time - start_time >= 2.0


@pytest.mark.asyncio
async def test_get_questions_batch_rate_limiting(
    qa_controller, test_file_path, mock_questions, mocker
):
    # Setup mock
    mock_get_response = mocker.patch(
        "src.qa_gpt.core.controller.qa_controller.get_chat_gpt_response_structure_async"
    )
    mock_get_response.return_value = mock_questions

    # Test with multiple files and fields
    file_paths = [test_file_path, test_file_path, test_file_path]
    field_names = ["field1", "field2", "field3"]
    field_values = ["value1", "value2", "value3"]

    start_time = time.time()
    results = await qa_controller.get_questions_batch(file_paths, field_names, field_values)
    end_time = time.time()

    # Assertions
    assert len(results) == 3
    assert all(isinstance(result, MultipleChoiceQuestionSet) for result in results)
    assert mock_get_response.call_count == 6  # 2 calls per file/field combination
    # Check that the total time is at least 2 seconds (3 calls with 1 second delay between each)
    assert end_time - start_time >= 2.0


@pytest.mark.asyncio
async def test_async_error_handling(qa_controller, test_file_path, mocker):
    # Setup mock to raise an exception
    mock_get_response = mocker.patch(
        "src.qa_gpt.core.controller.qa_controller.get_chat_gpt_response_structure_async"
    )
    mock_get_response.side_effect = Exception("Test error")

    # Test error handling in get_summary
    with pytest.raises(Exception) as exc_info:
        await qa_controller.get_summary(test_file_path, StandardSummary)
    assert str(exc_info.value) == "Test error"

    # Test error handling in get_questions
    with pytest.raises(Exception) as exc_info:
        await qa_controller.get_questions(test_file_path, "test_field", "test_value")
    assert str(exc_info.value) == "Test error"
