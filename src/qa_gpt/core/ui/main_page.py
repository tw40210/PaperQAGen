import streamlit as st

from src.qa_gpt.core.ui.materials_overview import display_materials_overview
from src.qa_gpt.core.ui.page_display import display_qa_page


def display_main_page():
    """
    Main entry point for the application with navigation between pages.
    """
    # Set wide mode
    st.set_page_config(layout="wide")

    # Add navigation header
    st.markdown(
        """
        <style>
        .nav-header {
            display: flex;
            justify-content: center;
            margin-bottom: 2rem;
        }
        .nav-button {
            margin: 0 1rem;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            text-decoration: none;
            color: white;
            background-color: #4CAF50;
        }
        .nav-button:hover {
            background-color: #45a049;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # Initialize session state for page selection if not exists
    if "current_page" not in st.session_state:
        st.session_state.current_page = "qa"

    # Navigation header
    st.markdown('<div class="nav-header">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Materials Overview", use_container_width=True):
            st.session_state.current_page = "materials"

    with col2:
        if st.button("QA Interface", use_container_width=True):
            st.session_state.current_page = "qa"

    st.markdown("</div>", unsafe_allow_html=True)

    # Display the selected page
    if st.session_state.current_page == "qa":
        display_qa_page()
    else:
        display_materials_overview()


if __name__ == "__main__":
    display_main_page()
