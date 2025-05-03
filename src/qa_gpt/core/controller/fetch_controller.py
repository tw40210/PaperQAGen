from pathlib import Path

from src.qa_gpt.core.controller.db_controller import (
    LocalDatabaseController,
    MaterialController,
)
from src.qa_gpt.core.controller.parsing_controller import ParsingController
from src.qa_gpt.core.controller.qa_controller import QAController
from src.qa_gpt.core.controller.rag_controller import RAGController
from src.qa_gpt.core.objects.summaries import (  # InnovationSummary,; TechnicalSummary,
    MetaDataSummary,
    StandardSummary,
)
from src.qa_gpt.core.utils.fetch_utils import (
    _filter_material_table_by_file_id,
    _should_skip_field_processing,
)
from src.qa_gpt.core.utils.pdf_processor import process_pdf_file


class FetchController:
    """Controller for fetching and processing materials."""

    def __init__(self):
        """Initialize the fetch controller."""
        self.db_name = "my_local_db"
        self.archive_name = "my_archive"
        self.local_db_controller = LocalDatabaseController(db_name=self.db_name)
        self.material_controller = MaterialController(
            db_controller=self.local_db_controller, archive_name=self.archive_name
        )
        self.qa_controller = QAController()
        self.parsing_controller = ParsingController()
        self.summary_objects = [
            StandardSummary,
            # TechnicalSummary,
            # InnovationSummary,
            MetaDataSummary,
        ]

    async def fetch_material_add_sets(self, file_id: str | None = None, process_all: bool = False):
        """Fetch material and add question sets to each material.

        Args:
            file_id: ID of a specific file to process. If None, will process all files.
            process_all: Must be set to True to process all files when file_id is None.
        """
        if file_id is None and not process_all:
            raise ValueError("Must set process_all=True to process all files when file_id is None")

        # Fetch material folder first
        self.material_controller.fetch_material_folder(Path("./pdf_data"))
        material_table = self.material_controller.get_material_table()

        # Filter material table if specific file_id is provided
        material_table = _filter_material_table_by_file_id(material_table, file_id)

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

            for summary_idx, summary_object in enumerate(self.summary_objects, 1):
                # Skip if summary type doesn't exist
                summary_type = summary_object.__name__
                if (
                    summary_type not in file_meta.summaries
                    or file_meta.summaries[summary_type] is None
                ):
                    print(f"Skipping {summary_type} as it doesn't exist.")
                    continue

                print(
                    f"\nProcessing summary {summary_idx}/{len(self.summary_objects )}: {summary_type}"
                )
                # Get the summary object
                summary = file_meta.summaries[summary_type]
                summary_dict = summary.model_dump()

                # Get questions for each top-level attribute
                total_fields = len(summary_dict)
                excluded_fields = set(summary_object.excluded_fields())
                for field_idx, (field_name, field_value) in enumerate(summary_dict.items(), 1):
                    # Create prefix for the question set
                    prefix = f"{summary_type}_{field_name}"

                    should_skip, reason = _should_skip_field_processing(
                        field_name, excluded_fields, file_meta, prefix, field_idx, total_fields
                    )
                    if should_skip:
                        print(reason)
                        continue

                    print(f"Processing field {field_idx}/{total_fields}: {field_name}")
                    file_paths.append(file_meta["file_path"])
                    field_names.append(field_name)
                    field_values.append(field_value)
                    prefixes.append(prefix)

            # Process all questions in batch
            if file_paths:
                question_sets = await self.qa_controller.get_questions_batch(
                    file_paths, field_names, field_values
                )
                for prefix, question_set in zip(prefixes, question_sets):
                    self.material_controller.append_mc_question_set(file_id, question_set, prefix)
                    print(f"Added question set for {prefix}")

            print(
                f"\nCompleted processing material {material_idx}/{total_materials} (ID: {file_id})"
            )

    async def fetch_material_add_summary(
        self, file_id: str | None = None, process_all: bool = False
    ):
        """Fetch material and add summary to each material.

        Args:
            file_id: ID of a specific file to process. If None, will process all files.
            process_all: Must be set to True to process all files when file_id is None.
        """
        if file_id is None and not process_all:
            raise ValueError("Must set process_all=True to process all files when file_id is None")

        # Fetch material folder first
        self.material_controller.fetch_material_folder(Path("./pdf_data"))
        material_table = self.material_controller.get_material_table()

        # Filter material table if specific file_id is provided
        material_table = _filter_material_table_by_file_id(material_table, file_id)

        total_materials = len(material_table)
        for material_idx, (file_id, file_meta) in enumerate(material_table.items(), 1):
            print(f"\nProcessing material {material_idx}/{total_materials} (ID: {file_id})")

            # Prepare batch processing data
            file_paths = []
            summary_classes = []
            summary_types = []

            for summary_idx, summary_object in enumerate(self.summary_objects, 1):
                # Skip if summary type already exists
                summary_type = summary_object.__name__
                if (
                    summary_type in file_meta.summaries
                    and file_meta.summaries[summary_type] is not None
                ):
                    print(f"Skipping {summary_type} as it already exists.")
                    continue

                print(
                    f"Processing summary {summary_idx}/{len(self.summary_objects )}: {summary_type}"
                )
                file_paths.append(file_meta["file_path"])
                summary_classes.append(summary_object)
                summary_types.append(summary_type)

            # Process all summaries in batch
            if len(file_paths) > 0:
                summaries = await self.qa_controller.get_summaries_batch(
                    file_paths, summary_classes
                )
                for summary_type, summary in zip(summary_types, summaries):
                    self.material_controller.append_summary(file_id, summary)
                    print(f"Added summary for {summary_type}")

            print(
                f"\nCompleted processing material {material_idx}/{total_materials} (ID: {file_id})"
            )

    def output_question_data(self, file_id: str | None = None, process_all: bool = False):
        """Output question data to a folder.

        Args:
            file_id: ID of a specific file to process. If None, will process all files.
            process_all: Must be set to True to process all files when file_id is None.
        """
        if file_id is None and not process_all:
            raise ValueError("Must set process_all=True to process all files when file_id is None")

        output_folder_path = Path("./output_question_data")
        output_folder_path.mkdir(exist_ok=True)

        # Fetch material folder first
        self.material_controller.fetch_material_folder(Path("./pdf_data"))
        material_table = self.material_controller.get_material_table()

        # Filter material table if specific file_id is provided
        material_table = _filter_material_table_by_file_id(material_table, file_id)

        self.material_controller.output_material_as_folder(output_folder_path)

    async def fetch_material_add_parsing(
        self, file_id: str | None = None, process_all: bool = False
    ):
        """Fetch material and add parsing results to each material.

        Args:
            file_id: ID of a specific file to process. If None, will process all files.
            process_all: Must be set to True to process all files when file_id is None.
        """
        if file_id is None and not process_all:
            raise ValueError("Must set process_all=True to process all files when file_id is None")

        # Create markdown folder if it doesn't exist
        markdown_folder = Path("./markdown")
        markdown_folder.mkdir(exist_ok=True)

        # Fetch material folder first
        self.material_controller.fetch_material_folder(Path("./pdf_data"))
        material_table = self.material_controller.get_material_table()

        # Filter material table if specific file_id is provided
        material_table = _filter_material_table_by_file_id(material_table, file_id)

        total_materials = len(material_table)
        for material_idx, (file_id, file_meta) in enumerate(material_table.items(), 1):
            print(f"\nProcessing material parsing {material_idx}/{total_materials} (ID: {file_id})")

            # Skip if parsing results already exist
            if file_meta["parsing_results"]["sections"] is not None:
                print(f"Skipping {file_id} as parsing results already exist.")
                continue

            # Process PDF to markdown first
            markdown_path = process_pdf_file(str(file_meta["file_path"]), str(markdown_folder))

            if markdown_path is None:
                print(f"Failed to process PDF to markdown for {file_id}")
                continue

            # Get parsing results from markdown file
            print(f"Processing markdown file for {file_id}")
            sections_object, images, tables = self.parsing_controller.get_sections_from_text_file(
                str(markdown_path)
            )

            # Update parsing results
            file_meta["parsing_results"] = {
                "sections": sections_object.sections,
                "images": images,
                "tables": tables,
            }

            # Save updated file meta
            target_path = self.material_controller.db_controller.get_target_path(
                [self.material_controller.db_table_name, str(file_id)]
            )
            self.material_controller.db_controller.save_data(file_meta, target_path)

            print(f"Added parsing results for material {file_id}")

        print(f"\nCompleted processing {total_materials} materials")

    async def build_rag_index(self, file_id: str | None = None, process_all: bool = False):
        """Build RAG index for parsed sections of materials.

        Args:
            file_id: ID of a specific file to process. If None, will process all files.
            process_all: Must be set to True to process all files when file_id is None.
        """
        if file_id is None and not process_all:
            raise ValueError("Must set process_all=True to process all files when file_id is None")

        # Fetch material folder first
        self.material_controller.fetch_material_folder(Path("./pdf_data"))
        material_table = self.material_controller.get_material_table()

        # Filter material table if specific file_id is provided
        material_table = _filter_material_table_by_file_id(material_table, file_id)

        total_materials = len(material_table)
        for material_idx, (file_id, file_meta) in enumerate(material_table.items(), 1):
            print(f"\nProcessing material RAG {material_idx}/{total_materials} (ID: {file_id})")

            # Skip if parsing results don't exist
            if file_meta["parsing_results"]["sections"] is None:
                print(f"Skipping {file_id} as parsing results don't exist.")
                continue

            # Skip if RAG state already exists
            if file_meta.rag_state is not None and file_meta.rag_state.exists():
                print(f"Skipping {file_id} as RAG state already exists.")
                continue

            try:
                # Initialize RAG controller
                rag_controller = RAGController(file_id=file_id)

                # Add sections to RAG index
                sections = file_meta["parsing_results"]["sections"]
                texts = [str(section) for section in sections]
                rag_controller.add_texts(texts)

                # Save RAG state
                rag_controller.save_state(rag_controller.state_path)

                # Update file meta with RAG state path
                file_meta.rag_state = rag_controller.state_path

                # Save updated file meta
                target_path = self.material_controller.db_controller.get_target_path(
                    [self.material_controller.db_table_name, str(file_id)]
                )
                self.material_controller.db_controller.save_data(file_meta, target_path)

                print(f"Added RAG index for material {file_id}")
            except Exception as e:
                print(f"Error processing {file_id}: {str(e)}")
                # Clean up any partially created state file
                if rag_controller.state_path.exists():
                    rag_controller.state_path.unlink()
                continue

        print(f"\nCompleted processing {total_materials} materials")
