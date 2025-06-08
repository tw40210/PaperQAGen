import asyncio

import streamlit as st

from src.qa_gpt import ONLY_DISPLAY
from src.qa_gpt.core.ui.material_operations import display_material_operations
from src.qa_gpt.core.ui.material_selection import display_material_selection
from src.qa_gpt.core.ui.question_display import (
    display_question_results,
    display_questions,
    load_questions,
)
from src.qa_gpt.core.ui.summary_display import display_summary
from src.qa_gpt.core.utils.parsing_utils import extract_question_set_id


def display_qa_page(folder_path: str = "output_question_data"):
    """
    Display the main page layout with all components.

    Args:
        folder_path (str): Path to the folder containing question data
    """
    _, col1, _, col2, _ = st.columns([1, 4, 1, 6, 1])

    with col1:
        # Add file upload functionality
        if ONLY_DISPLAY:
            st.file_uploader("Upload a PDF file", type=["pdf"], disabled=True)
        else:
            from src.qa_gpt.core.ui.file_upload import handle_file_upload

            asyncio.run(handle_file_upload())

        # Display material selection and get selected material and file
        material_folder_path, selected_file = display_material_selection(folder_path)

        # Display material operations
        if selected_file is not None:
            display_material_operations(
                material_folder_path, question_set_id=extract_question_set_id(selected_file)
            )

        # Display questions if a file is selected
        if selected_file:
            file_path = f"{material_folder_path}/{selected_file}"
            questions = load_questions(file_path)
            user_selections = display_questions(
                questions,
                material_folder_path=material_folder_path,
                question_set_id=extract_question_set_id(selected_file),
            )
            display_question_results(user_selections)

    with col2:
        # Display summary
        display_summary(material_folder_path)
