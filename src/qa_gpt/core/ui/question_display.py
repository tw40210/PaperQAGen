import json
from typing import Any

import streamlit as st

from src.qa_gpt.core.utils.display_utils import display_question, get_parsed_question


def load_questions(file_path: str) -> dict[str, Any]:
    """Load questions from a JSON file."""
    with open(file_path) as file:
        return json.load(file)


def display_questions(
    questions: dict[str, Any]
) -> list[tuple[str, list[int], list[int], list[str]]]:
    """Display questions and collect user selections.

    Args:
        questions: Dictionary containing question data

    Returns:
        List of tuples containing:
        - Question description
        - Selected options
        - Correct answers
        - Explanations
    """
    st.header("Question set")
    st.write("---")

    user_selections = []

    for question_key, question_set in questions.items():
        parsed_question = get_parsed_question(question_set)
        selected_options = display_question(
            parsed_question["question_description"], parsed_question["options"], question_key
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
