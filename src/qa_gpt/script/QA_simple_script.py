from pathlib import Path

from src.qa_gpt.core.utils.fetch_utils import initialize_controllers_and_get_file_id


def process_material(file_path: Path | str) -> None:
    """Process a material file to generate questions and summaries.

    Args:
        file_path: Path to the material file to process
    """
    try:
        # Initialize controllers and get file ID
        material_controller, file_id = initialize_controllers_and_get_file_id(file_path)

        # Process the material
        material_controller.fetch_material_folder(Path(file_path).parent)
        material_controller.process_material(file_id)

    except Exception as e:
        print(f"Error processing material: {str(e)}")
        raise


if __name__ == "__main__":
    # Example usage - replace with your actual file path
    material_path = Path("pdf_data/case_study_o1_29.pdf")
    process_material(material_path)
