import streamlit as st

from src.qa_gpt.core.objects.summaries import (
    InnovationSummary,
    StandardSummary,
    TechnicalSummary,
)


def display_summary_beautifully(
    data: dict | StandardSummary | TechnicalSummary | InnovationSummary, title: str = None
) -> None:
    """
    Display summary data in a beautiful and organized way using Streamlit.

    Args:
        data: The summary data to display
        title: Optional title to display above the summary
    """
    if title:
        st.subheader(title)

    # Create a container for better organization
    with st.container():
        # Handle different summary types
        if isinstance(data, dict):
            summary_type = data.get("summary_type", "")
            if summary_type == "standard":
                _display_standard_summary(data)
            elif summary_type == "technical":
                _display_technical_summary(data)
            elif summary_type == "innovation":
                _display_innovation_summary(data)
            else:
                _display_generic_json(data)
        else:
            _display_generic_json(data)


def _display_standard_summary(data: dict) -> None:
    """Display a standard summary with motivation, conclusion, and bullet points."""
    # Display Motivation
    st.markdown("### 🎯 Motivation")
    motivation = data.get("motivation", {})
    st.markdown(f"**Description:** {motivation.get('description', '')}")
    st.markdown(f"**Problem to Solve:** {motivation.get('problem_to_solve', '')}")
    st.markdown(f"**How to Solve:** {motivation.get('how_to_solve', '')}")
    st.markdown(f"**Why Can Be Solved:** {motivation.get('why_can_be_solved', '')}")

    st.markdown("---")

    # Display Conclusion
    st.markdown("### 🏁 Conclusion")
    conclusion = data.get("conclusion", {})
    st.markdown(f"**Description:** {conclusion.get('description', '')}")
    st.markdown(f"**Problem Solved:** {conclusion.get('problem_to_solve', '')}")
    st.markdown(f"**How Much Solved:** {conclusion.get('how_much_is_solved', '')}")
    st.markdown(f"**Contribution:** {conclusion.get('contribution', '')}")

    st.markdown("---")

    # Display Bullet Points
    st.markdown("### 📌 Key Points")
    bullet_points = data.get("bullet_points", [])
    for idx, point in enumerate(bullet_points, 1):
        with st.expander(f"Point {idx}: {point.get('subject', '')}", expanded=True):
            st.markdown(f"**Description:** {point.get('description', '')}")
            st.markdown(f"**Technical Details:** {point.get('technical_details', '')}")
            st.markdown(f"**Importance:** {point.get('importance_explanation', '')}")
            importance = point.get("importance", 0)
            st.markdown(f"**Importance Level:** {'⭐' * importance}")


def _display_technical_summary(data: dict) -> None:
    """Display a technical summary with overview, concepts, and details."""
    st.markdown("### 📋 Overview")
    st.markdown(data.get("overview", ""))

    st.markdown("### 🔑 Key Concepts")
    for concept in data.get("key_concepts", []):
        st.markdown(f"- {concept}")

    st.markdown("### ⚙️ Technical Details")
    for detail in data.get("technical_details", []):
        st.markdown(f"- {detail}")

    st.markdown("### 📝 Implementation Steps")
    for step in data.get("implementation_steps", []):
        st.markdown(f"- {step}")

    st.markdown("### 📋 Requirements")
    for req in data.get("requirements", []):
        st.markdown(f"- {req}")

    st.markdown("### ⚠️ Limitations")
    for limit in data.get("limitations", []):
        st.markdown(f"- {limit}")


def _display_innovation_summary(data: dict) -> None:
    """Display an innovation summary with overview, concepts, and innovation points."""
    st.markdown("### 📋 Overview")
    st.markdown(data.get("overview", ""))

    st.markdown("### 🔑 Key Concepts")
    for concept in data.get("key_concepts", []):
        st.markdown(f"- {concept}")

    st.markdown("### 💡 Innovation Points")
    innovation_points = data.get("innovation_points", [])
    for idx, point in enumerate(innovation_points, 1):
        with st.expander(f"Point {idx}: {point.get('subject', '')}", expanded=True):
            st.markdown(f"**Description:** {point.get('description', '')}")
            st.markdown(f"**Technical Details:** {point.get('technical_details', '')}")
            st.markdown(f"**Why Innovative:** {point.get('why_is_innovative', '')}")
            st.markdown("**Similar Concepts:**")
            for concept in point.get("similar_concepts", []):
                st.markdown(f"- {concept}")
            importance = point.get("importance", 0)
            st.markdown(f"**Importance Level:** {'⭐' * importance}")

    st.markdown("### 📚 References")
    for ref in data.get("references", []):
        st.markdown(f"- {ref}")


def _display_generic_json(data: dict | list) -> None:
    """Display generic JSON data in a formatted way."""
    with st.expander("View Data", expanded=True):
        if isinstance(data, dict):
            for key, value in data.items():
                st.markdown(f"**{key}:**")
                if isinstance(value, (dict, list)):
                    st.json(value)
                else:
                    st.write(value)
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                st.markdown(f"**Item {idx + 1}:**")
                st.json(item)
        else:
            st.json(data)


def get_parsed_question(question_object) -> dict:
    parsed_question = {}
    choices = ["choice_1", "choice_2", "choice_3", "choice_4"]
    parsed_question["question_description"] = question_object["question_description"]
    parsed_question["options"] = [
        question_object[choice]["choice_description"] for choice in choices
    ]
    parsed_question["answers"] = [
        idx for idx, choice in enumerate(choices) if question_object[choice]["answer"]
    ]
    parsed_question["explanation"] = [question_object[choice]["explanation"] for choice in choices]

    return parsed_question


def display_question(question, options, question_key):
    st.write(question)
    selected_options = []

    for idx, option in enumerate(options):
        if st.checkbox(option, key=f"{question_key}_{option}"):
            selected_options.append(idx)

    return selected_options
