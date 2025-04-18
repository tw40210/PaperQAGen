import os
import shutil
from pathlib import Path

import streamlit as st

from src.qa_gpt.core.controller.db_controller import MaterialController
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
    material_folder_path: str | None, material_controller: MaterialController | None = None
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

    operation = st.selectbox("Select an operation", ["Remove selected material"])

    if st.button("Run Operation"):
        if operation == "Remove selected material":
            remove_material(material_folder_path, material_controller)
