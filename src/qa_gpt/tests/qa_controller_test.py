import asyncio
from unittest.mock import MagicMock, patch

import pytest

from src.qa_gpt.core.controller.qa_controller import QAController
from src.qa_gpt.core.controller.rag_controller import RAGController
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
def test_file_id():
    return "test_file_123"


@pytest.fixture
def qa_controller():
    controller = QAController()
    return controller


@pytest.fixture
def mock_rag_controller():
    controller = MagicMock(spec=RAGController)
    controller.search_text.return_value = [("Test content", 0.1)]
    return controller


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


def test_get_summary(qa_controller, test_file_id, mock_summary, mocker, mock_rag_controller):
    # Setup mock
    mock_get_response = mocker.patch(
        "src.qa_gpt.core.controller.qa_controller.get_chat_gpt_response_structure_async"
    )
    mock_get_response.return_value = mock_summary

    with patch("src.qa_gpt.core.controller.qa_controller.RAGController.from_file_id") as mock_rag:
        mock_rag.return_value = mock_rag_controller

        # Test
        result = asyncio.run(qa_controller.get_summary(test_file_id, StandardSummary))

        # Assertions
        assert isinstance(result, StandardSummary)
        assert result == mock_summary
        mock_get_response.assert_called_once()
        mock_rag.assert_called_once_with(test_file_id)
        mock_rag.return_value.search_text.assert_called_once_with(
            "base_summary, conclusion, findings, results, contribution, solved, outcome, achievement, impact, significance, future_work, limitations, recommendations",
            k=5,
        )


def test_get_questions(qa_controller, test_file_id, mock_questions, mocker, mock_rag_controller):
    # Setup mock
    mock_get_response = mocker.patch(
        "src.qa_gpt.core.controller.qa_controller.get_chat_gpt_response_structure_async"
    )
    mock_get_response.return_value = mock_questions

    with patch("src.qa_gpt.core.controller.qa_controller.RAGController.from_file_id") as mock_rag:
        mock_rag.return_value = mock_rag_controller

        # Test
        result = asyncio.run(qa_controller.get_questions(test_file_id, "test_field", "test_value"))

        # Assertions
        assert isinstance(result, MultipleChoiceQuestionSet)
        assert result == mock_questions
        assert (
            mock_get_response.call_count == 1
        )  # Only one call to get_chat_gpt_response_structure_async
        assert mock_rag.call_count == 1  # Only one call to RAGController.from_file_id
        mock_rag.return_value.search_text.assert_called_with(
            "test_field", k=2
        )  # k=2 as per implementation


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


@pytest.mark.asyncio
async def test_get_summary_async(
    qa_controller, test_file_id, mock_summary, mocker, mock_rag_controller
):
    # Setup mock
    mock_get_response = mocker.patch(
        "src.qa_gpt.core.controller.qa_controller.get_chat_gpt_response_structure_async"
    )
    mock_get_response.return_value = mock_summary

    with patch("src.qa_gpt.core.controller.qa_controller.RAGController.from_file_id") as mock_rag:
        mock_rag.return_value = mock_rag_controller

        # Test
        result = await qa_controller.get_summary(test_file_id, StandardSummary)

        # Assertions
        assert isinstance(result, StandardSummary)
        assert result == mock_summary
        mock_get_response.assert_called_once()
        mock_rag.assert_called_once_with(test_file_id)
        mock_rag.return_value.search_text.assert_called_once_with(
            "base_summary, conclusion, findings, results, contribution, solved, outcome, achievement, impact, significance, future_work, limitations, recommendations",
            k=5,
        )


@pytest.mark.asyncio
async def test_get_questions_async(
    qa_controller, test_file_id, mock_questions, mocker, mock_rag_controller
):
    # Setup mock
    mock_get_response = mocker.patch(
        "src.qa_gpt.core.controller.qa_controller.get_chat_gpt_response_structure_async"
    )
    mock_get_response.return_value = mock_questions

    with patch("src.qa_gpt.core.controller.qa_controller.RAGController.from_file_id") as mock_rag:
        mock_rag.return_value = mock_rag_controller

        # Test
        result = await qa_controller.get_questions(test_file_id, "test_field", "test_value")

        # Assertions
        assert isinstance(result, MultipleChoiceQuestionSet)
        assert result == mock_questions
        assert (
            mock_get_response.call_count == 1
        )  # Only one call to get_chat_gpt_response_structure_async
        assert mock_rag.call_count == 1  # Only one call to RAGController.from_file_id
        mock_rag.return_value.search_text.assert_called_with(
            "test_field", k=2
        )  # k=2 as per implementation


@pytest.mark.asyncio
async def test_get_questions_batch_rate_limiting(
    qa_controller, test_file_id, mock_questions, mocker, mock_rag_controller
):
    # Setup mock
    mock_get_response = mocker.patch(
        "src.qa_gpt.core.controller.qa_controller.get_chat_gpt_response_structure_async"
    )
    mock_get_response.return_value = mock_questions

    with patch("src.qa_gpt.core.controller.qa_controller.RAGController.from_file_id") as mock_rag:
        mock_rag.return_value = mock_rag_controller

        # Test with multiple files and fields
        file_ids = [test_file_id, test_file_id, test_file_id]
        field_names = ["field1", "field2", "field3"]
        field_values = ["value1", "value2", "value3"]

        results = await qa_controller.get_questions_batch(file_ids, field_names, field_values)

        # Assertions
        assert len(results) == 3
        assert all(isinstance(result, MultipleChoiceQuestionSet) for result in results)
        assert mock_get_response.call_count == 3  # One call per file/field combination
        assert mock_rag.call_count == 3  # One call per file/field combination
        for field_name in field_names:
            mock_rag.return_value.search_text.assert_any_call(
                field_name, k=2
            )  # k=2 as per implementation


@pytest.mark.asyncio
async def test_async_error_handling(qa_controller, test_file_id, mocker, mock_rag_controller):
    # Setup mock to raise an exception
    mock_get_response = mocker.patch(
        "src.qa_gpt.core.controller.qa_controller.get_chat_gpt_response_structure_async"
    )
    mock_get_response.side_effect = Exception("Test error")

    with patch("src.qa_gpt.core.controller.qa_controller.RAGController.from_file_id") as mock_rag:
        mock_rag.return_value = mock_rag_controller

        # Test error handling in get_summary
        with pytest.raises(Exception) as exc_info:
            await qa_controller.get_summary(test_file_id, StandardSummary)
        assert str(exc_info.value) == "Test error"

        # Test error handling in get_questions
        with pytest.raises(Exception) as exc_info:
            await qa_controller.get_questions(test_file_id, "test_field", "test_value")
        assert str(exc_info.value) == "Test error"
