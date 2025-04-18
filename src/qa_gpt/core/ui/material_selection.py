import os

import streamlit as st


def get_display_name(folder_name: str) -> str:
    """Extract display name from folder name by removing the ID suffix.

    Args:
        folder_name: Full folder name in format "{file_name}_{id}"

    Returns:
        Display name without the ID suffix
    """
    return folder_name.rsplit("_", 1)[0]


def get_material_folders(folder_path: str, search_query: str) -> tuple[list[str], dict[str, str]]:
    """Get filtered material folders based on search query.

    Args:
        folder_path: Path to the folder containing materials
        search_query: Search query to filter materials

    Returns:
        Tuple containing:
        - List of display names (without IDs)
        - Dictionary mapping display names to full folder names
    """
    material_folders = list(os.listdir(folder_path))
    display_to_full = {
        get_display_name(folder): folder
        for folder in material_folders
        if search_query.lower() in folder.lower()
    }
    return list(display_to_full.keys()), display_to_full


def get_question_files(material_folder_path: str) -> list[str]:
    """Get question files from a material folder."""
    return [
        f
        for f in os.listdir(material_folder_path)
        if f.endswith(".json") and not f.startswith("meta_data") and not f.startswith("summary")
    ]


def get_summary_files(material_folder_path: str) -> list[str]:
    """Get summary files from a material folder."""
    return [f for f in os.listdir(material_folder_path) if f.startswith("summary")]


def display_material_selection(folder_path: str) -> tuple[str | None, str | None]:
    """Display material selection UI and return selected material and file.

    Args:
        folder_path: Path to the folder containing materials

    Returns:
        Tuple containing:
        - Selected material folder path (or None if none selected)
        - Selected file name (or None if none selected)
    """
    st.header("Material selection")

    search_query = st.text_input("Search material", "")
    display_names, display_to_full = get_material_folders(folder_path, search_query)
    selected_display = st.selectbox("Select a material folder", display_names)

    selected_file = None
    material_folder_path = None

    if selected_display:
        full_folder_name = display_to_full[selected_display]
        material_folder_path = os.path.join(folder_path, full_folder_name)
        files = get_question_files(material_folder_path)
        selected_file = st.selectbox("Select a file", files)

    return material_folder_path, selected_file
