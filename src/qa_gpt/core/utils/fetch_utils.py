from pathlib import Path

from src.qa_gpt.core.controller.db_controller import (
    LocalDatabaseController,
    MaterialController,
)
from src.qa_gpt.core.controller.qa_controller import QAController
from src.qa_gpt.core.objects.summaries import (
    InnovationSummary,
    MetaDataSummary,
    StandardSummary,
    TechnicalSummary,
)

summary_objects = [StandardSummary, TechnicalSummary, InnovationSummary, MetaDataSummary]


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


async def fetch_material_add_sets(file_id: str | None = None, process_all: bool = False):
    """Fetch material and add question sets to each material.

    Args:
        file_id: ID of a specific file to process. If None, will process all files.
        process_all: Must be set to True to process all files when file_id is None.
    """
    if file_id is None and not process_all:
        raise ValueError("Must set process_all=True to process all files when file_id is None")

    material_controller = initialize_controllers()
    qa_controller = QAController()

    # Fetch material folder first
    material_controller.fetch_material_folder(Path("./pdf_data"))
    material_table = material_controller.get_material_table()

    # Filter material table if specific file_id is provided
    if file_id is not None:
        if file_id not in material_table:
            raise ValueError(f"File ID {file_id} not found in material table")
        material_table = {file_id: material_table[file_id]}

    total_materials = len(material_table)
    for material_idx, (file_id, file_meta) in enumerate(material_table.items(), 1):
        print(f"\nProcessing material {material_idx}/{total_materials} (ID: {file_id})")
        print(
            f"Material {file_id} originally have {len(file_meta.mc_question_sets)} mc_questions sets."
        )

        # Prepare batch processing data
        file_paths = []
        field_names = []
        field_values = []
        prefixes = []

        for summary_idx, summary_object in enumerate(summary_objects, 1):
            # Skip if summary type doesn't exist
            summary_type = summary_object.__name__
            if summary_type not in file_meta.summaries or file_meta.summaries[summary_type] is None:
                print(f"Skipping {summary_type} as it doesn't exist.")
                continue

            print(f"\nProcessing summary {summary_idx}/{len(summary_objects)}: {summary_type}")
            # Get the summary object
            summary = file_meta.summaries[summary_type]
            summary_dict = summary.model_dump()

            # Get questions for each top-level attribute
            total_fields = len(summary_dict)
            excluded_fields = set(summary_object.excluded_fields())
            for field_idx, (field_name, field_value) in enumerate(summary_dict.items(), 1):
                if field_name in excluded_fields:
                    print(f"Excluding field {field_idx}/{total_fields}: {field_name}")
                    continue
                print(f"Processing field {field_idx}/{total_fields}: {field_name}")
                # Create prefix for the question set
                prefix = f"{summary_type}_{field_name}"

                # Count existing question sets with this prefix
                existing_count = sum(
                    1 for key in file_meta.mc_question_sets.keys() if key.startswith(prefix)
                )
                if existing_count > 0:
                    print(
                        f"Skipping {prefix} as {existing_count} question set(s) already exist(s)."
                    )
                    continue

                file_paths.append(file_meta["file_path"])
                field_names.append(field_name)
                field_values.append(field_value)
                prefixes.append(prefix)

        # Process all questions in batch
        if file_paths:
            question_sets = await qa_controller.get_questions_batch(
                file_paths, field_names, field_values
            )
            for prefix, question_set in zip(prefixes, question_sets):
                material_controller.append_mc_question_set(file_id, question_set, prefix)
                print(f"Added question set for {prefix}")

        print(f"\nCompleted processing material {material_idx}/{total_materials} (ID: {file_id})")


async def fetch_material_add_summary(file_id: str | None = None, process_all: bool = False):
    """Fetch material and add summary to each material.

    Args:
        file_id: ID of a specific file to process. If None, will process all files.
        process_all: Must be set to True to process all files when file_id is None.
    """
    if file_id is None and not process_all:
        raise ValueError("Must set process_all=True to process all files when file_id is None")

    material_controller = initialize_controllers()
    qa_controller = QAController()

    # Fetch material folder first
    material_controller.fetch_material_folder(Path("./pdf_data"))
    material_table = material_controller.get_material_table()

    # Filter material table if specific file_id is provided
    if file_id is not None:
        if file_id not in material_table:
            raise ValueError(f"File ID {file_id} not found in material table")
        material_table = {file_id: material_table[file_id]}

    total_materials = len(material_table)
    for material_idx, (file_id, file_meta) in enumerate(material_table.items(), 1):
        print(f"\nProcessing material {material_idx}/{total_materials} (ID: {file_id})")

        # Prepare batch processing data
        file_paths = []
        summary_classes = []
        summary_types = []

        for summary_idx, summary_object in enumerate(summary_objects, 1):
            # Skip if summary type already exists
            summary_type = summary_object.__name__
            if (
                summary_type in file_meta.summaries
                and file_meta.summaries[summary_type] is not None
            ):
                print(f"Skipping {summary_type} as it already exists.")
                continue

            print(f"Processing summary {summary_idx}/{len(summary_objects)}: {summary_type}")
            file_paths.append(file_meta["file_path"])
            summary_classes.append(summary_object)
            summary_types.append(summary_type)

        # Process all summaries in batch
        if file_paths:
            summaries = await qa_controller.get_summaries_batch(file_paths, summary_classes)
            for summary_type, summary in zip(summary_types, summaries):
                material_controller.append_summary(file_id, summary)
                print(f"Added summary for {summary_type}")

        print(f"\nCompleted processing material {material_idx}/{total_materials} (ID: {file_id})")


def output_question_data(file_id: str | None = None, process_all: bool = False):
    """Output question data to a folder.

    Args:
        file_id: ID of a specific file to process. If None, will process all files.
        process_all: Must be set to True to process all files when file_id is None.
    """
    if file_id is None and not process_all:
        raise ValueError("Must set process_all=True to process all files when file_id is None")

    material_controller = initialize_controllers()
    output_folder_path = Path("./output_question_data")
    output_folder_path.mkdir(exist_ok=True)

    # Fetch material folder first
    material_controller.fetch_material_folder(Path("./pdf_data"))
    material_table = material_controller.get_material_table()

    # Filter material table if specific file_id is provided
    if file_id is not None:
        if file_id not in material_table:
            raise ValueError(f"File ID {file_id} not found in material table")

    material_controller.output_material_as_folder(output_folder_path)
