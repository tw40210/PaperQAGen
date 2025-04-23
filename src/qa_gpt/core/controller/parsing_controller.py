from src.qa_gpt.chat.chat import get_chat_gpt_response_structure
from src.qa_gpt.core.objects.parsing import TextSections
from src.qa_gpt.core.utils.pdf_processor import extract_markdown_elements


class ParsingController:
    """Controller for parsing and analyzing text files"""

    def __init__(self) -> None:
        """Initialize the parsing controller"""
        self.system_prompt = """
        You are a markdown text analysis assistant. Your task is to break down the given markdown text into logical sections while following these rules:

        1. Preserve the original text exactly as it appears in the input
        2. Keep the content field focused on the main text flow
        3. Provide a clear title and summary for each section
        4. Maintain the original markdown formatting in the content field
        5. Ensure sections are coherent and maintain the original meaning
        """

    def get_sections_from_text_file(self, file_path: str) -> TextSections:
        """
        Analyze a text file and break it down into structured sections using OpenAI.

        Args:
            file_path: Path to the text file to analyze

        Returns:
            TextSections: Structured analysis of the text with sections
        """
        # Read the text file
        with open(file_path, encoding="utf-8") as f:
            text_content = f.read()
            images, tables, text_content = extract_markdown_elements(text_content)

        # Prepare the prompt for OpenAI
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"Please analyze the following markdown text and break it down into sections:\n\n{text_content}",
            },
        ]

        # Get structured response from OpenAI
        sections = get_chat_gpt_response_structure(messages, TextSections)
        return sections, images, tables

    def save_sections_to_file(self, analysis: TextSections, output_path: str) -> None:
        """
        Save the text analysis to a markdown file.

        Args:
            analysis: The text analysis to save
            output_path: Path where to save the analysis
        """
        with open(output_path, "w", encoding="utf-8") as f:
            for section in analysis.sections:
                f.write(f"# {section.title}\n\n")
                f.write(f"**Summary:** {section.summary}\n\n")
                f.write(f"{section.content}\n\n")
                f.write("---\n\n")
