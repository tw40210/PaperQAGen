#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

from src.qa_gpt.core.utils.pdf_processor import process_pdf_directory, process_pdf_file


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def main():
    parser = argparse.ArgumentParser(description="Process PDF files and convert them to Markdown")
    parser.add_argument("input", help="Input PDF file or directory containing PDF files")
    parser.add_argument("output", help="Output directory for Markdown files")
    parser.add_argument(
        "--single", action="store_true", help="Process a single file instead of a directory"
    )

    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return

    if args.single:
        if not input_path.is_file():
            logger.error(f"Input path must be a file when using --single: {input_path}")
            return
        if not input_path.suffix.lower() == ".pdf":
            logger.error(f"Input file must be a PDF: {input_path}")
            return

        result = process_pdf_file(str(input_path), str(output_path))
        if result:
            logger.info(f"Successfully processed {input_path} to {result}")
        else:
            logger.error(f"Failed to process {input_path}")
    else:
        if not input_path.is_dir():
            logger.error(f"Input path must be a directory when not using --single: {input_path}")
            return

        results = process_pdf_directory(str(input_path), str(output_path))
        logger.info(f"Processed {len(results)} PDF files")
        for result in results:
            logger.info(f"Generated: {result}")


if __name__ == "__main__":
    main()
