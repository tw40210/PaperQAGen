import json
from pathlib import Path

import streamlit as st

from src.qa_gpt.core.controller.fetch_controller import FetchController
from src.qa_gpt.core.utils.fetch_utils import initialize_controllers_and_get_file_id


def load_json_from_file(file_path):
    """Load JSON data from a file."""
    with open(file_path) as file:
        return json.load(file)


async def handle_file_upload():
    """Handle file upload with validation and process using fetch functions."""
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Check if file is PDF
        if not uploaded_file.name.lower().endswith(".pdf"):
            st.error("Please upload a PDF file")
            return

        # Create pdf_data directory if it doesn't exist
        pdf_data_path = Path("./pdf_data")
        pdf_data_path.mkdir(exist_ok=True)

        # Check for duplicate file names
        file_path = pdf_data_path / uploaded_file.name
        if file_path.exists():
            st.error(f"A file with name '{uploaded_file.name}' already exists")
            return

        # Save the file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Show warning about processing time
        st.warning("⚠️ Processing the file might take several minutes. Please wait...")

        try:
            # Get material controller and file ID
            material_controller, file_id = initialize_controllers_and_get_file_id(file_path)

            # Process the uploaded file using FetchController
            with st.spinner("Processing the uploaded file..."):
                # Initialize FetchController with the existing material controller
                fetch_controller = FetchController()

                await fetch_controller.fetch_material_add_parsing(file_id=file_id)
                await fetch_controller.build_rag_index(file_id=file_id)
                await fetch_controller.fetch_material_add_summary(file_id=file_id)
                await fetch_controller.fetch_material_add_sets(file_id=file_id)
                fetch_controller.output_question_data(file_id=file_id)

            st.success(f"File '{uploaded_file.name}' uploaded and processed successfully")
            # Reset the file uploader state
            st.session_state.file_uploader_key = None
        except ValueError as e:
            st.error(str(e))
            return
