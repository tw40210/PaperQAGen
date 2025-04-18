from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.qa_gpt.core.controller.qa_controller import QAController
from src.qa_gpt.core.objects.questions import MultipleChoiceQuestionSet
from src.qa_gpt.core.objects.summaries import StandardSummary


@pytest.mark.asyncio
async def test_get_summary():
    test_file_path = Path("./test_data/test_input_1.pdf")

    qa_controller = QAController()
    # Mock the _pdf_to_text method to avoid file operations
    qa_controller.preprocess_controller._pdf_to_text = MagicMock(return_value="Test PDF content")
    preprocess_result = qa_controller.preprocess_controller.preprocess(test_file_path)
    assert preprocess_result is not None

    result = await qa_controller.get_summary(test_file_path, StandardSummary)
    assert result is not None
    assert isinstance(result, StandardSummary)


@pytest.mark.asyncio
async def test_get_questions():
    test_file_path = Path("./test_data/test_input_2.pdf")

    qa_controller = QAController()
    preprocess_result = qa_controller.preprocess_controller.preprocess(test_file_path)
    assert preprocess_result is not None

    result = await qa_controller.get_questions(test_file_path, "test_field", "test_value")
    assert result is not None
    assert isinstance(result, MultipleChoiceQuestionSet)


if __name__ == "__main__":
    test_get_questions()
