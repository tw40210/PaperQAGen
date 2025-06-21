import os

import streamlit as st

from src.qa_gpt.core.utils.fetch_utils import initialize_controllers
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

    # Initialize material controller to access database
    material_controller = initialize_controllers()

    # Walk through the directory
    for root, _, files in os.walk(folder_path):
        # Find the material in the database by matching the folder name
        folder_name = os.path.basename(root)
        file_name = folder_name.rsplit("_", 1)[0]  # Remove the ID suffix
        # Get material metadata from database
        file_meta = material_controller.get_material_by_filename(file_name)

        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                question_set_id = extract_question_set_id(file)

                # Get relative path from the base folder
                rel_path = os.path.relpath(root, folder_path)

                # Get comment statistics from database using material controller
                total_comments = 0
                positive_comments = 0

                if file_meta:
                    question_comments = file_meta.question_comments
                    # Filter to only keep comments for the current question_set_id
                    question_comments = {
                        k: v
                        for k, v in question_comments.items()
                        if k.startswith(f"{question_set_id}_")
                    }
                    total_comments = len(question_comments)
                    positive_comments = sum(
                        1 for comment in question_comments.values() if comment.is_positive
                    )

                if question_set_id.startswith("meta_data") or question_set_id.startswith("summary"):
                    continue
                else:
                    # Handle case when file_meta is None
                    paper_title = "Unknown"
                    if file_meta and "MetaDataSummary" in file_meta.summaries:
                        paper_title = file_meta.summaries["MetaDataSummary"].paper_title

                    materials.append(
                        {
                            "question_set_name": question_set_id,
                            "material_name": rel_path.split("_")[0],
                            "file": file,
                            "full_path": file_path,
                            "paper_title": paper_title,
                            "total_comments": total_comments,
                            "positive_comments": positive_comments,
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

    # Create a table with material information including comment statistics
    st.dataframe(
        data=[
            {
                "Material Name": m["material_name"],
                "Paper Title": m["paper_title"],
                "Question Set Name": m["question_set_name"],
                "Total Comments": m["total_comments"],
                "Positive Comments": m["positive_comments"],
                "Positive %": (
                    f"{(m['positive_comments'] / m['total_comments'] * 100):.1f}%"
                    if m["total_comments"] > 0
                    else "N/A"
                ),
            }
            for m in materials
        ],
        use_container_width=True,
        hide_index=True,
    )
