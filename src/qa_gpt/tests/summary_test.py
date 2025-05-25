from unittest.mock import MagicMock, patch

import pytest

from src.qa_gpt.core.controller.qa_controller import QAController
from src.qa_gpt.core.controller.rag_controller import RAGController
from src.qa_gpt.core.objects.summaries import (
    BulletPoint,
    Conclusion,
    Motivation,
    StandardSummary,
    TechnicalSummary,
)


@pytest.fixture
def mock_rag_controller():
    controller = MagicMock(spec=RAGController)
    controller.search_text.return_value = [("Test content", 0.1)]
    return controller


@pytest.fixture
def qa_controller():
    return QAController()


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
async def test_get_standard_summary(qa_controller, mock_rag_controller):
    test_file_id = "test_file_123"

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
async def test_get_technical_summary(qa_controller, mock_rag_controller):
    test_file_id = "test_file_123"

    with patch("src.qa_gpt.core.controller.qa_controller.RAGController.from_file_id") as mock_rag:
        mock_rag.return_value = mock_rag_controller

        result = await qa_controller.get_summary(test_file_id, TechnicalSummary)
        assert result is not None
        assert isinstance(result, TechnicalSummary)
        mock_rag.assert_called_once_with(test_file_id)
        mock_rag_controller.search_text.assert_called_once_with(
            "technical, implementation, architecture, requirements, specifications, algorithms, data_structures, performance, optimization, maintenance, deployment, configuration, dependencies, limitations, constraints, metrics",
            k=5,
        )


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
