from pydantic import BaseModel, Field


class TextSection(BaseModel):
    """Represents a section of text with its content and summary"""

    title: str = Field(..., description="Title of the section")
    content: str = Field(
        ..., description="Content of the section, should be a clip from the original text"
    )
    summary: str = Field(..., description="Brief summary of the section")


class TextSections(BaseModel):
    """Represents the complete analysis of a text with multiple sections"""

    sections: list[TextSection] = Field(..., description="List of text sections")
