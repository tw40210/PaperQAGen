from pathlib import Path

import PyPDF2


def pdf_to_text(pdf_path: Path):

    # Open the PDF file in read-binary mode
    with open(str(pdf_path), "rb") as pdf_file:
        # Create a PdfReader object instead of PdfFileReader
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # Initialize an empty string to store the text
        text = ""

        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()

    return text


def extract_question_set_id(file_name: str) -> str:
    """Extract question set ID from a question file name.

    Args:
        file_name: The name of the question file (e.g. 'mc_question_StandardSummary_conclusion_0.json')

    Returns:
        str: The question set ID (e.g. 'StandardSummary_conclusion_0')

    Examples:
        >>> extract_question_set_id('mc_question_StandardSummary_conclusion_0.json')
        'StandardSummary_conclusion_0'
        >>> extract_question_set_id('mc_question_0.json')
        '0'
    """
    # Remove the file extension
    name_without_ext = file_name.rsplit(".", 1)[0]

    # Remove the 'mc_question_' prefix
    if name_without_ext.startswith("mc_question_"):
        return name_without_ext[len("mc_question_") :]

    return name_without_ext
