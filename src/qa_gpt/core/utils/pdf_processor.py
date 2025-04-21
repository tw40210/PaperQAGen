import logging
from pathlib import Path

from magic_pdf.data.data_reader_writer import FileBasedDataReader, FileBasedDataWriter
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze

logger = logging.getLogger(__name__)


class PDFProcessor:
    """PDF processor: Converts PDF to Markdown format"""

    def __init__(self):
        """
        Initialize PDF processor
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.debug("Initializing PDF processor")

    def process(self, pdf_path: str, output_dir: str) -> Path:
        """
        Process PDF file

        Args:
            pdf_path: Path to PDF file
            output_dir: Path to output directory

        Returns:
            Path: Path to generated Markdown file

        Raises:
            FileNotFoundError: When PDF file does not exist
        """
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

        try:
            # Set output paths
            paper_name = pdf_path.stem
            output_image_path = output_dir / "images"
            local_image_path = "images"

            # Initialize writers
            image_writer = FileBasedDataWriter(str(output_image_path))
            md_writer = FileBasedDataWriter(str(output_dir))

            # Read PDF file
            reader = FileBasedDataReader("")
            pdf_bytes = reader.read(pdf_path)  # Read PDF content

            # Create dataset instance
            ds = PymuDocDataset(pdf_bytes)

            # Process PDF
            self.logger.info("Starting PDF processing...")
            ds.apply(doc_analyze, ocr=True).pipe_ocr_mode(image_writer).dump_md(
                md_writer, f"{paper_name}.md", local_image_path
            )

            # Generate Markdown path
            markdown_path = output_dir / f"{paper_name}.md"

            self.logger.info(f"Markdown file saved to: {markdown_path}")
            return markdown_path

        except Exception as e:
            self.logger.error(f"PDF processing failed: {str(e)}", exc_info=True)
            raise


def process_pdf_file(pdf_path: str, output_dir: str) -> Path | None:
    """
    Process a single PDF file and convert it to Markdown

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save the output

    Returns:
        Optional[Path]: Path to the generated Markdown file if successful, None otherwise
    """
    try:
        processor = PDFProcessor()
        return processor.process(pdf_path, output_dir)
    except Exception as e:
        logger.error(f"Failed to process PDF {pdf_path}: {str(e)}")
        return None


def process_pdf_directory(input_dir: str, output_dir: str) -> list[Path]:
    """
    Process all PDF files in a directory and convert them to Markdown

    Args:
        input_dir: Directory containing PDF files
        output_dir: Directory to save the output files

    Returns:
        list[Path]: List of paths to generated Markdown files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    processed_files = []

    for pdf_file in input_path.glob("*.pdf"):
        try:
            result = process_pdf_file(str(pdf_file), str(output_path))
            if result:
                processed_files.append(result)
        except Exception as e:
            logger.error(f"Failed to process {pdf_file}: {str(e)}")
            continue

    return processed_files
