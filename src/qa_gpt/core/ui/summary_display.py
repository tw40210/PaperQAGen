import json

import streamlit as st

from src.qa_gpt.core.utils.display_utils import display_summary_beautifully


def load_summary(file_path: str) -> dict:
    """Load summary from a JSON file."""
    with open(file_path) as file:
        return json.load(file)


def display_summary(material_folder_path: str | None) -> None:
    """Display the material summary.

    Args:
        material_folder_path: Path to the material folder (or None if no material selected)
    """
    st.header("Material summary")

    if not material_folder_path:
        st.write("No material selected.")
        return

    from src.qa_gpt.core.ui.material_selection import get_summary_files

    summary_files = get_summary_files(material_folder_path)

    if not summary_files:
        st.write("No summary files found for this material.")
        return

    selected_summary = st.selectbox("Select summary type", summary_files)
    if selected_summary:
        summary_path = f"{material_folder_path}/{selected_summary}"
        summary = load_summary(summary_path)
        display_summary_beautifully(summary, "Material Summary")
