import json
from pathlib import Path
from typing import Any

import streamlit as st

from src.qa_gpt.core.controller.db_controller import MaterialController
from src.qa_gpt.core.utils.display_utils import display_question, get_parsed_question
from src.qa_gpt.core.utils.fetch_utils import initialize_controllers


def load_questions(file_path: str) -> dict[str, Any]:
    """Load questions from a JSON file."""
    with open(file_path) as file:
        return json.load(file)


def display_questions(
    questions: dict[str, Any],
    material_folder_path: str | None = None,
    question_set_id: str | None = None,
    material_controller: MaterialController | None = None,
) -> list[tuple[str, list[int], list[int], list[str]]]:
    """Display questions and collect user selections.

    Args:
        questions: Dictionary containing question data
        material_folder_path: Optional path to the material folder
        question_set_id: Optional ID of the question set
        material_controller: Optional MaterialController instance

    Returns:
        List of tuples containing:
        - Question description
        - Selected options
        - Correct answers
        - Explanations
    """
    st.header("Question set")
    st.write("---")

    # Initialize material controller and get file meta if material folder path is provided
    file_meta = None
    if material_folder_path:
        # Initialize material controller if not provided
        if material_controller is None:
            material_controller = initialize_controllers()

        # Get the file name from the folder path
        folder_name = Path(material_folder_path).name
        file_name = folder_name.rsplit("_", 1)[0]  # Remove the ID suffix

        # Get the file meta for the selected material
        file_meta = material_controller.get_material_by_filename(file_name)

    user_selections = []

    for question_key, question_set in questions.items():
        parsed_question = get_parsed_question(question_set)

        selected_options = display_question(
            parsed_question["question_description"],
            parsed_question["options"],
            question_key,
            question_set_id,
            material_controller,
            file_meta,
        )
        user_selections.append(
            (
                parsed_question["question_description"],
                selected_options,
                parsed_question["answers"],
                parsed_question["explanation"],
            )
        )
        st.write("---")

    return user_selections


def display_question_results(
    user_selections: list[tuple[str, list[int], list[int], list[str]]]
) -> None:
    """Display the results of the user's answers.

    Args:
        user_selections: List of tuples containing question data and user selections
    """
    if st.button("Submit"):
        for question, selected, answers, explanations in user_selections:
            st.write(f"**{question}**")
            st.write(f"Your selection : {selected}")
            st.write(f"Correct answers : {answers}")
            st.write("Explanation : \n")
            for i in range(len(explanations)):
                st.write(f"Option {i} is {i in answers}. {explanations[i]}\n")

            if set(selected) == set(answers):
                st.success("All correct!")
            else:
                st.error("Some options wrong.")
            st.write("---")
