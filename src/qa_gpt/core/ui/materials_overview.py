import os

import streamlit as st

from src.qa_gpt.core.utils.parsing_utils import extract_question_set_id


def get_all_materials(folder_path: str = "output_question_data") -> list[dict]:
    """
    Get all available materials and their metadata.

    Args:
        folder_path (str): Path to the folder containing question data

    Returns:
        List[Dict]: List of materials with their metadata
    """
    materials = []

    # Walk through the directory
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                question_set_id = extract_question_set_id(file)

                # Get relative path from the base folder
                rel_path = os.path.relpath(root, folder_path)

                materials.append(
                    {
                        "name": question_set_id,
                        "path": rel_path,
                        "file": file,
                        "full_path": file_path,
                    }
                )

    return materials


def display_materials_overview(folder_path: str = "output_question_data"):
    """
    Display a table of all available materials.

    Args:
        folder_path (str): Path to the folder containing question data
    """
    st.title("Materials Overview")

    # Get all materials
    materials = get_all_materials(folder_path)

    if not materials:
        st.info("No materials found. Please upload some materials first.")
        return

    # Create a table with material information
    st.dataframe(
        data=[
            {"Material Name": m["name"], "Location": m["path"], "File": m["file"]}
            for m in materials
        ],
        use_container_width=True,
        hide_index=True,
    )
