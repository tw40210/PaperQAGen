import shutil
from pathlib import Path

from src.qa_gpt.core.constant import LOCAL_DB_FOLDER
from src.qa_gpt.core.controller.db_controller import (
    LocalDatabaseController,
    MaterialController,
)


def test_local_db():
    test_db_name = "test_local_db"
    test_local_db_controller = LocalDatabaseController(db_name=test_db_name)
    save_path = "material_1.questionSet_0"
    save_data = {"q": "QQ", "a": "AA"}
    test_local_db_controller.save_data(save_data, save_path)

    assert save_data == test_local_db_controller.get_data(save_path)

    update_data = {"c": "QQ", "a": "AA", "q": "QAQ"}
    test_local_db_controller.update_data(update_data, save_path)

    assert update_data == test_local_db_controller.get_data(save_path)

    test_local_db_controller.delete_data(save_path)

    assert test_local_db_controller.get_data(save_path) is None

    test_local_db_controller.db_path.unlink()


def test_local_material_controller_input():
    test_db_name = "test_local_db"
    test_archive_name = "test_archive"
    test_source_folder_path = Path("./test_data")

    # Clean up any existing archive directory
    archive_path = Path(f"archived_materials/{test_archive_name}")
    if archive_path.exists():
        shutil.rmtree(archive_path)

    # Clean up any existing database files
    db_path = Path(f"{LOCAL_DB_FOLDER}/{test_db_name}.pkl")
    if db_path.exists():
        db_path.unlink()

    # Ensure test data directory exists
    if not test_source_folder_path.exists():
        test_source_folder_path.mkdir(exist_ok=True)

    test_local_db_controller = LocalDatabaseController(db_name=test_db_name)
    test_material_controller = MaterialController(
        db_controller=test_local_db_controller, archive_name=test_archive_name
    )

    test_material_controller.fetch_material_folder(test_source_folder_path)

    # Get the material table and mapping table
    material_table = test_material_controller.get_material_table()
    mapping_table = test_material_controller.get_material_mapping_table()

    # Verify that IDs in mapping table are strings
    for file_name, material_id in mapping_table.items():
        assert isinstance(material_id, str), f"Material ID for {file_name} should be a string"

    assert len(material_table) == len(list(test_source_folder_path.iterdir()))
    assert len(list(test_material_controller.archive_path.iterdir())) == len(
        list(test_source_folder_path.iterdir())
    )

    # Clean up
    shutil.rmtree(test_material_controller.archive_path)
    test_local_db_controller.db_path.unlink()


def test_local_material_controller_output():
    test_db_name = "test_local_output_db"
    test_archive_name = "test_archive"
    test_output_folder_path = Path("./test_output_question_data")

    # Clean up any existing output directory
    if test_output_folder_path.exists():
        shutil.rmtree(test_output_folder_path)
    test_output_folder_path.mkdir(exist_ok=True)

    # Clean up any existing database files
    db_path = Path(f"{LOCAL_DB_FOLDER}/{test_db_name}.pkl")
    if db_path.exists():
        db_path.unlink()

    test_local_db_controller = LocalDatabaseController(db_name=test_db_name)
    test_material_controller = MaterialController(
        db_controller=test_local_db_controller, archive_name=test_archive_name
    )
    test_material_controller.output_material_as_folder(test_output_folder_path)

    assert len(test_material_controller.get_material_table()) == len(
        list(test_output_folder_path.iterdir())
    )

    # Clean up
    shutil.rmtree(test_output_folder_path)
    if db_path.exists():
        db_path.unlink()


if __name__ == "__main__":
    test_local_db()
    test_local_material_controller_input()
    test_local_material_controller_output()
