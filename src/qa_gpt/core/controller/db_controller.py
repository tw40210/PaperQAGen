import json
import logging
import pickle
import shutil
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path

from src.qa_gpt.core.constant import LOCAL_DB_FOLDER, MATERIAL_FOLDER
from src.qa_gpt.core.objects.materials import FileMeta
from src.qa_gpt.core.objects.questions import MultipleChoiceQuestionSet
from src.qa_gpt.core.objects.summaries import StandardSummary, TechnicalSummary

logger = logging.getLogger(__name__)


class BasicDatabaseController(ABC):
    @abstractmethod
    def __init__(self, db_name: str):
        pass

    @abstractmethod
    def get_data(self, target_path: str) -> any:
        pass

    @abstractmethod
    def save_data(self, data: dict, target_path: str) -> int:
        pass

    @abstractmethod
    def delete_data(self, target_path: str) -> int:
        pass

    @abstractmethod
    def update_data(self, data: str, target_path: str) -> int:
        pass


class LocalDatabaseController(BasicDatabaseController):
    def __init__(self, db_name: str = "local_db") -> None:
        self.db_folder_path = Path(f"{LOCAL_DB_FOLDER}")
        self.db_path = Path(f"{LOCAL_DB_FOLDER}/{db_name}.pkl")
        self.db = {}
        self.db_folder_path.mkdir(exist_ok=True)

        self._init_local_df()

    def _init_local_df(self) -> None:
        if self.db_path.exists():
            with open(str(self.db_path), "rb") as db_file:
                self.db = pickle.load(db_file)

    def _commit(self) -> int:
        with open(str(self.db_path), "wb") as db_file:
            pickle.dump(self.db, db_file)

        return 0

    def _query_path(self, target_path: str, create_path: bool = False):
        prev = None
        leaf_key = None
        cur = self.db
        for key in target_path.split("."):
            if key not in cur:
                if create_path:
                    cur[key] = {}
                else:
                    return {}, None, leaf_key
            prev = cur
            leaf_key = key
            cur = cur[key]
        return prev, cur, leaf_key

    def get_data(self, target_path: str) -> any:
        logger.debug(f"Try to get `{target_path}`.")

        prev, cur, leaf_key = self._query_path(target_path)

        return cur

    def save_data(self, data: dict, target_path: str) -> int:
        prev, _, leaf_key = self._query_path(target_path, create_path=True)
        prev[leaf_key] = data

        self._commit()
        logger.debug(f"`{target_path}` is newly saved.")

        return 0

    def delete_data(self, target_path: str) -> int:
        prev, _, leaf_key = self._query_path(target_path)
        if leaf_key in prev:
            del prev[leaf_key]
            self._commit()
            logger.debug(f"`{target_path}` is deleted.")
        else:
            logger.warning(f"`{target_path}` is not found. Nothing deleted.")

        return 0

    def update_data(self, data: str, target_path: str) -> int:
        self.delete_data(target_path)
        self.save_data(data, target_path)

        self._commit()
        logger.debug(f"`{target_path}` is updated.")

        return 0

    @staticmethod
    def get_target_path(path_list: list[str]) -> str:
        return ".".join(path_list)


class MaterialController:
    def __init__(self, db_controller: BasicDatabaseController, archive_name: str) -> None:
        self.db_controller = db_controller
        self.material_folder_path = Path(MATERIAL_FOLDER)
        self.archive_path = Path(f"{MATERIAL_FOLDER}/{archive_name}")
        self.db_table_name = "material_table"
        self.db_mapping_table_name = "material_id_mapping_table"
        self.material_folder_path.mkdir(exist_ok=True)
        self.archive_path.mkdir(exist_ok=True)

        # init in db
        if self.db_controller.get_data(self.db_table_name) is None:
            self.db_controller.save_data({}, self.db_table_name)
        if self.db_controller.get_data(self.db_mapping_table_name) is None:
            self.db_controller.save_data({}, self.db_mapping_table_name)

    @staticmethod
    def remove_dot_from_file_name(file_path: Path) -> Path:
        # We don't allow "." in file names since it's used in db query
        new_file_name = file_path.stem.replace(".", "_") + file_path.suffix
        new_file_path = file_path.parent / Path(new_file_name)

        return new_file_path

    def fetch_material_folder(self, source_folder_path: Path):
        archive_file_id = len(self.db_controller.get_data(self.db_mapping_table_name))

        for file_path in sorted(source_folder_path.iterdir()):
            if (
                not str(file_path).endswith(".pdf")
                or self.db_controller.get_data(
                    LocalDatabaseController.get_target_path(
                        [self.db_mapping_table_name, file_path.stem]
                    )
                )
                is not None
            ):
                continue

            # Rename the file in case there is an illegal name.
            new_file_path = MaterialController.remove_dot_from_file_name(file_path)
            file_path.rename(new_file_path)
            file_path = new_file_path

            file_meta = FileMeta(
                id=archive_file_id,
                file_name=file_path.stem,
                file_suffix=file_path.suffix,
                file_path=self.archive_path
                / Path(f"{file_path.stem}_{archive_file_id}{file_path.suffix}"),
                mc_question_sets={},
                summaries={},
                parsing_results={"sections": None, "images": None, "tables": None},
            )

            db_path = LocalDatabaseController.get_target_path(
                [self.db_table_name, str(archive_file_id)]
            )
            db_mapping_path = LocalDatabaseController.get_target_path(
                [self.db_mapping_table_name, file_meta["file_name"]]
            )

            self.db_controller.save_data(file_meta, db_path)
            self.db_controller.save_data(str(archive_file_id), db_mapping_path)

            shutil.copy(file_path, file_meta["file_path"])

            logger.info(f"{file_path.stem} is archived with id:{archive_file_id}.")

            archive_file_id += 1

    def _get_material_filemeta(self, file_id: int) -> FileMeta:
        target_path = LocalDatabaseController.get_target_path([self.db_table_name, str(file_id)])
        file_meta = self.db_controller.get_data(target_path)
        return file_meta, target_path

    def append_mc_question_set(
        self, file_id: int, question_set: MultipleChoiceQuestionSet, prefix: str = ""
    ) -> int:
        file_meta, target_path = self._get_material_filemeta(file_id)
        mc_question_sets = file_meta["mc_question_sets"]

        # Count existing question sets with the same prefix
        existing_prefix_count = sum(1 for key in mc_question_sets.keys() if key.startswith(prefix))

        # Create a unique ID by combining prefix with count
        question_set_id = (
            f"{prefix}_{existing_prefix_count}" if prefix else str(len(mc_question_sets))
        )

        file_meta["mc_question_sets"][question_set_id] = question_set
        self.db_controller.save_data(file_meta, target_path)

        return 0

    def append_summary(self, file_id: int, summary: StandardSummary | TechnicalSummary) -> int:
        file_meta, target_path = self._get_material_filemeta(file_id)
        summary_type = summary.__class__.__name__
        file_meta["summaries"][summary_type] = summary

        self.db_controller.save_data(file_meta, target_path)

        return 0

    def get_material_table(self) -> dict[str, FileMeta]:
        return self.db_controller.get_data(self.db_table_name)

    def get_material_mapping_table(self) -> dict:
        return self.db_controller.get_data(self.db_mapping_table_name)

    def remove_material_by_filename(self, file_name: str) -> int:
        """Remove a material and its associated data by file name.

        Args:
            file_name: The name of the file to remove (without extension)

        Returns:
            int: 0 if successful, -1 if material not found

        Raises:
            FileNotFoundError: If the physical file cannot be deleted
        """
        # Get the material ID from mapping table
        mapping_table = self.get_material_mapping_table()
        if file_name not in mapping_table:
            logger.warning(f"Material with file name '{file_name}' not found")
            return -1

        material_id = mapping_table[file_name]

        # Get the material metadata
        material_table = self.get_material_table()
        if str(material_id) not in material_table:
            logger.warning(f"Material with ID {material_id} not found in material table")
            return -1

        file_meta = material_table[str(material_id)]

        # Delete the physical file
        try:
            file_meta["file_path"].unlink()
        except FileNotFoundError:
            logger.warning(f"Physical file not found at {file_meta['file_path']}")

        # Remove from material table
        self.db_controller.delete_data(
            LocalDatabaseController.get_target_path([self.db_table_name, str(material_id)])
        )

        # Remove from mapping table
        self.db_controller.delete_data(
            LocalDatabaseController.get_target_path([self.db_mapping_table_name, file_name])
        )

        logger.info(f"Successfully removed material '{file_name}' with ID {material_id}")
        return 0

    def output_material_as_folder(self, output_folder_path: Path):
        material_table = self.get_material_table()
        material_ids = list(material_table.keys())

        for material_id in material_ids:
            material_folder = output_folder_path / Path(
                material_table[str(material_id)].file_path.stem
            )
            material_folder.mkdir(exist_ok=True)
            meta_file_path = material_folder / Path("meta_data.json")
            summary_file_path = material_folder / Path("summary.json")

            meta_dict = asdict(material_table[str(material_id)])
            meta_dict["file_path"] = str(meta_dict["file_path"])
            meta_dict.pop("mc_question_sets")
            meta_dict.pop("summaries")
            meta_dict.pop("parsing_results")

            mc_question_sets = material_table[str(material_id)].mc_question_sets
            summaries = material_table[str(material_id)].summaries

            with open(str(meta_file_path), "w") as file:
                json.dump(meta_dict, file, indent=4)

            for summary_type, summary in summaries.items():
                summary_file_path = material_folder / Path(f"summary_{summary_type}.json")
                with open(str(summary_file_path), "w") as file:
                    json.dump(summary.model_dump(), file, indent=4)

            for set_id, mc_question_set in mc_question_sets.items():
                mc_question_file_path = material_folder / Path(f"mc_question_{set_id}.json")
                with open(str(mc_question_file_path), "w") as file:
                    json.dump(mc_question_set.model_dump(), file, indent=4)

        return 0
