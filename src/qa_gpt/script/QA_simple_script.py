from src.qa_gpt.core.utils.fetch_utils import (
    fetch_material_add_sets,
    fetch_material_add_summary,
    output_question_data,
)

if __name__ == "__main__":
    fetch_material_add_summary()
    fetch_material_add_sets()
    output_question_data()
