from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from src.qa_gpt.core.objects.questions import MultipleChoiceQuestionSet
from src.qa_gpt.core.objects.summaries import StandardSummary, TechnicalSummary


class BasicDataBaseObject(ABC):

    @abstractmethod
    def __getitem__(self, key):
        pass

    @abstractmethod
    def __setitem__(self, key, value):
        pass

    @abstractmethod
    def __delitem__(self, key):
        pass

    @abstractmethod
    def __contains__(self, key):
        pass


@dataclass
class FileMeta(BasicDataBaseObject):
    id: int
    file_name: str
    file_suffix: str
    file_path: Path
    mc_question_sets: dict[str, MultipleChoiceQuestionSet]
    summaries: dict[str, StandardSummary | TechnicalSummary]
    parsing_results: dict[str, any]
    rag_state: Path | None = None  # Path to the RAG state file

    def __getitem__(self, key):
        # Get the attribute using dictionary-like syntax
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Set the attribute using dictionary-like syntax
        if key not in self.__dict__:
            raise ValueError(f"Unexpected attribute key: {key}")
        setattr(self, key, value)

    def __delitem__(self, key):
        # Delete the attribute using dictionary-like syntax
        delattr(self, key)

    def __contains__(self, key):
        # Check if an attribute exists using `in`
        return hasattr(self, key)
