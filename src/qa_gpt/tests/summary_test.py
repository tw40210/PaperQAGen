from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from src.qa_gpt.core.controller.qa_controller import QAController
from src.qa_gpt.core.objects.summaries import (
    BulletPoint,
    Conclusion,
    Motivation,
    StandardSummary,
    TechnicalSummary,
)


@pytest.fixture
def qa_controller():
    controller = QAController()
    # Mock the _pdf_to_text method to avoid file operations
    controller.preprocess_controller._pdf_to_text = MagicMock(return_value="Test PDF content")
    return controller


@pytest.fixture
def mock_standard_summary():
    return StandardSummary(
        motivation=Motivation(
            description="The motivation behind this material is to present information in a structured format that is easy to digest and analyze.",
            problem_to_solve="Test problem",
            how_to_solve="Test solution",
            why_can_be_solved="Test why",
        ),
        conclusion=Conclusion(
            description="Test conclusion",
            problem_to_solve="Test problem",
            how_much_is_solved="Test solution",
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
def mock_technical_summary():
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


@pytest.mark.asyncio
async def test_get_standard_summary(mock_standard_summary, qa_controller, mocker: MockerFixture):
    # Setup mock
    mock_get_response = mocker.patch(
        "src.qa_gpt.core.controller.qa_controller.get_chat_gpt_response_structure_async"
    )
    mock_get_response.return_value = mock_standard_summary

    # Test
    result = await qa_controller.get_summary(Path("test.pdf"), StandardSummary)

    # Assertions
    assert isinstance(result, StandardSummary)
    assert result == mock_standard_summary


@pytest.mark.asyncio
async def test_get_technical_summary(mock_technical_summary, qa_controller, mocker: MockerFixture):
    # Setup mock
    mock_get_response = mocker.patch(
        "src.qa_gpt.core.controller.qa_controller.get_chat_gpt_response_structure_async"
    )
    mock_get_response.return_value = mock_technical_summary

    # Test
    result = await qa_controller.get_summary(Path("test.pdf"), TechnicalSummary)

    # Assertions
    assert isinstance(result, TechnicalSummary)
    assert result == mock_technical_summary


def test_summary_serialization(mock_standard_summary, mock_technical_summary):
    # Test standard summary serialization
    standard_dict = mock_standard_summary.model_dump()
    assert "summary_type" in standard_dict
    assert standard_dict["summary_type"] == "standard"
    assert "motivation" in standard_dict
    assert "conclusion" in standard_dict
    assert "bullet_points" in standard_dict
    assert len(standard_dict["bullet_points"]) == 1

    # Test technical summary serialization
    technical_dict = mock_technical_summary.model_dump()
    assert "overview" in technical_dict
    assert "key_concepts" in technical_dict
    assert "technical_details" in technical_dict
    assert "implementation_steps" in technical_dict
    assert "requirements" in technical_dict
    assert "limitations" in technical_dict
