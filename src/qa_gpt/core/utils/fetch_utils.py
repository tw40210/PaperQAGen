from pathlib import Path

from src.qa_gpt.core.controller.db_controller import (
    LocalDatabaseController,
    MaterialController,
)


def initialize_controllers() -> MaterialController:
    """Initialize and return a MaterialController instance.

    Returns:
        MaterialController: An initialized MaterialController instance.
    """
    db_name = "my_local_db"
    archive_name = "my_archive"
    local_db_controller = LocalDatabaseController(db_name=db_name)
    return MaterialController(db_controller=local_db_controller, archive_name=archive_name)


def initialize_controllers_and_get_file_id(file_path: Path | str) -> tuple[MaterialController, str]:
    """Initialize controllers and get file ID for a given file path.

    Args:
        file_path: Path to the file to process.

    Returns:
        Tuple of (material_controller, file_id)

    Raises:
        ValueError: If file ID cannot be found for the given file.
    """
    # Convert to Path if string
    if isinstance(file_path, str):
        file_path = Path(file_path)

    # Initialize controllers
    material_controller = initialize_controllers()

    # Fetch the material folder to get the file ID
    material_controller.fetch_material_folder(file_path.parent)

    # Get the file ID from the mapping table
    file_name = file_path.stem
    mapping_path = LocalDatabaseController.get_target_path(
        [material_controller.db_mapping_table_name, file_name]
    )
    file_id = material_controller.db_controller.get_data(mapping_path)

    if file_id is None:
        raise ValueError(f"Failed to get file ID for file: {file_path}")

    return material_controller, file_id


def _filter_material_table_by_file_id(material_table: dict, file_id: str | None) -> dict:
    """Filter material table by file_id if provided.

    Args:
        material_table: The material table to filter
        file_id: ID of a specific file to process. If None, returns the original table.

    Returns:
        Filtered material table containing only the specified file_id if provided,
        otherwise returns the original table.

    Raises:
        ValueError: If file_id is provided but not found in material table.
    """
    if file_id is not None:
        if file_id not in material_table:
            raise ValueError(f"File ID {file_id} not found in material table")
        return {file_id: material_table[file_id]}
    return material_table


def _should_skip_field_processing(
    field_name: str,
    excluded_fields: set,
    file_meta: dict,
    prefix: str,
    field_idx: int,
    total_fields: int,
) -> tuple[bool, str | None]:
    """Check if we should skip processing a field.

    Args:
        field_name: Name of the field to check
        excluded_fields: Set of fields to exclude
        file_meta: File metadata containing question sets
        prefix: Prefix for the question set
        field_idx: Current field index
        total_fields: Total number of fields

    Returns:
        Tuple of (should_skip, reason) where:
        - should_skip: Boolean indicating whether to skip processing
        - reason: String explaining why we're skipping, or None if not skipping
    """
    if field_name in excluded_fields:
        return True, f"Excluding field {field_idx}/{total_fields}: {field_name}"

    # Count existing question sets with this prefix
    existing_count = sum(1 for key in file_meta.mc_question_sets.keys() if key.startswith(prefix))
    if existing_count > 0:
        return True, f"Skipping {prefix} as {existing_count} question set(s) already exist(s)."

    return False, None
