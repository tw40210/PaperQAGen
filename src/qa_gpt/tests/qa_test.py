from unittest.mock import MagicMock, patch

import pytest

from src.qa_gpt.core.controller.qa_controller import QAController
from src.qa_gpt.core.controller.rag_controller import RAGController
from src.qa_gpt.core.objects.questions import MultipleChoiceQuestionSet
from src.qa_gpt.core.objects.summaries import StandardSummary


@pytest.fixture
def mock_rag_controller():
    controller = MagicMock(spec=RAGController)
    controller.search_text.return_value = [("Test content", 0.1)]
    return controller


@pytest.mark.asyncio
async def test_get_summary(mock_rag_controller):
    test_file_id = "test_file_123"

    qa_controller = QAController()
    with patch("src.qa_gpt.core.controller.qa_controller.RAGController.from_file_id") as mock_rag:
        mock_rag.return_value = mock_rag_controller

        result = await qa_controller.get_summary(test_file_id, StandardSummary)
        assert result is not None
        assert isinstance(result, StandardSummary)
        mock_rag.assert_called_once_with(test_file_id)
        mock_rag_controller.search_text.assert_called_once_with(
            "base_summary, conclusion, findings, results, contribution, solved, outcome, achievement, impact, significance, future_work, limitations, recommendations",
            k=5,
        )


@pytest.mark.asyncio
async def test_get_questions(mock_rag_controller):
    test_file_id = "test_file_123"

    qa_controller = QAController()
    with patch("src.qa_gpt.core.controller.qa_controller.RAGController.from_file_id") as mock_rag:
        mock_rag.return_value = mock_rag_controller

        result = await qa_controller.get_questions(test_file_id, "test_field", "test_value")
        assert result is not None
        assert isinstance(result, MultipleChoiceQuestionSet)
        assert mock_rag.call_count == 1  # Only one call to RAGController.from_file_id
        mock_rag_controller.search_text.assert_called_with(
            "test_field", k=2
        )  # k=2 as per implementation


if __name__ == "__main__":
    test_get_questions()
