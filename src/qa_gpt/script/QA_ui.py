import streamlit as st

import asyncio



from src.qa_gpt.core.ui.file_upload import handle_file_upload
from src.qa_gpt.core.ui.material_operations import display_material_operations
from src.qa_gpt.core.ui.material_selection import display_material_selection
from src.qa_gpt.core.ui.question_display import (
    display_question_results,
    display_questions,
    load_questions,
)
from src.qa_gpt.core.ui.summary_display import display_summary

# Set wide mode
st.set_page_config(layout="wide")
_, col1, _, col2, _ = st.columns([1, 4, 1, 6, 1])

# Constants
FOLDER_PATH = "output_question_data"

with col1:
    # Add file upload functionality
    asyncio.run(handle_file_upload())

    # Display material selection and get selected material and file
    material_folder_path, selected_file = display_material_selection(FOLDER_PATH)

    # Display material operations
    display_material_operations(material_folder_path)

    # Display questions if a file is selected
    if selected_file:
        file_path = f"{material_folder_path}/{selected_file}"
        questions = load_questions(file_path)
        user_selections = display_questions(questions)
        display_question_results(user_selections)

with col2:
    # Display summary
    display_summary(material_folder_path)
