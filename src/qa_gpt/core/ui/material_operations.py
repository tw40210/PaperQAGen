import os
import shutil
from pathlib import Path

import streamlit as st

from src.qa_gpt import ONLY_DISPLAY
from src.qa_gpt.core.controller.db_controller import MaterialController
from src.qa_gpt.core.objects.materials import FileMeta
from src.qa_gpt.core.objects.questions import QuestionComment
from src.qa_gpt.core.utils.fetch_utils import initialize_controllers


def remove_material(
    material_folder_path: str, material_controller: MaterialController | None = None
) -> None:
    """Remove the selected material folder and all its contents.

    Args:
        material_folder_path: Path to the material folder to remove
        material_controller: Optional MaterialController instance. If None, a new one will be initialized.
    """
    if os.path.exists(material_folder_path):
        # Get the file name from the folder path
        folder_name = Path(material_folder_path).name
        file_name = folder_name.rsplit("_", 1)[0]  # Remove the ID suffix

        # Initialize material controller if not provided
        if material_controller is None:
            material_controller = initialize_controllers()

        # Remove from database and controller
        result = material_controller.remove_material_by_filename(file_name)

        if result == 0:
            # Remove the physical folder
            shutil.rmtree(material_folder_path)

            # Remove the original PDF file from pdf_data directory
            pdf_path = Path("./pdf_data") / f"{file_name}.pdf"
            if pdf_path.exists():
                pdf_path.unlink()

            st.success("Material removed successfully!")
            # Clear the selection and refresh the page
            st.session_state.clear()
            st.rerun()
        else:
            st.error("Failed to remove material from database")
    else:
        st.error("Material folder not found!")


def display_material_operations(
    material_folder_path: str | None,
    question_set_id: str | None,
    material_controller: MaterialController | None = None,
) -> None:
    """Display material operations UI.

    Args:
        material_folder_path: Path to the selected material folder (or None if no material selected)
        material_controller: Optional MaterialController instance. If None, a new one will be initialized.
    """
    st.header("Material Operations")

    if not material_folder_path:
        st.write("No material selected.")
        return

    operation = st.selectbox(
        "Select an operation", ["Comment on a question", "Remove selected material"]
    )

    if operation == "Remove selected material":
        if ONLY_DISPLAY:
            st.error("This operation is not available in display mode")
            return

        if st.button("Run Operation"):
            remove_material(material_folder_path, material_controller)
    elif operation == "Comment on a question":
        # Get the file name from the folder path
        folder_name = Path(material_folder_path).name
        file_name = folder_name.rsplit("_", 1)[0]  # Remove the ID suffix

        # Initialize material controller if not provided
        if material_controller is None:
            material_controller = initialize_controllers()

        # Get the file meta for the selected material
        file_meta = material_controller.get_material_by_filename(file_name)

        if file_meta:
            st.subheader("Add Question Comment")

            # Create a form for comment submission
            with st.form("comment_form", clear_on_submit=True):
                # Input fields for comment
                topic = st.text_input("Topic")
                content = st.text_area("Comment Content")
                is_positive = st.checkbox("Is this a positive comment?")

                # Question selection
                question_id = st.selectbox(
                    "Question ID",
                    ["question_1", "question_2", "question_3", "question_4", "question_5"],
                )

                # Submit button
                submitted = st.form_submit_button("Submit Comment")

                if submitted:
                    if topic and content and question_set_id:
                        add_question_comment(
                            file_meta,
                            topic,
                            content,
                            is_positive,
                            question_set_id,
                            question_id,
                            material_controller,
                        )
                    else:
                        st.error("Please fill in all required fields")
        else:
            st.error("Could not find material information")


def add_question_comment(
    file_meta: FileMeta,
    topic: str,
    content: str,
    is_positive: bool,
    question_set_id: str,
    question_id: str,
    material_controller: MaterialController | None = None,
) -> None:
    """Create and add a QuestionComment to the FileMeta object using the MaterialController.

    Args:
        file_meta (FileMeta): The FileMeta object to add the comment to
        topic (str): The topic of the comment
        content (str): The content of the comment
        is_positive (bool): Whether the comment is positive or negative
        question_set_id (str): The ID of the question set
        question_id (str): The ID of the question (must be 'question_1' through 'question_5')
        material_controller (MaterialController | None): Optional MaterialController instance. If None, a new one will be initialized.

    Returns:
        None

    Raises:
        ValueError: If a comment for the same topic already exists or if the question set/question doesn't exist
    """
    try:
        # Initialize material controller if not provided
        if material_controller is None:
            material_controller = initialize_controllers()

        comment = QuestionComment(
            topic=topic,
            content=content,
            is_positive=is_positive,
            question_set_id=question_set_id,
            question_id=question_id,
        )

        result = material_controller.append_question_comment(file_meta["id"], comment)

        if result == 0:
            st.success(f"Successfully added comment for topic: {topic}")
        else:
            st.error("Failed to add comment to database")

    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Failed to add comment: {str(e)}")
