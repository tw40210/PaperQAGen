import argparse
from pathlib import Path

from src.qa_gpt.core.controller.parsing_controller import ParsingController


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a text file and break it down into structured sections."
    )
    parser.add_argument("input_file", type=str, help="Path to the input text file")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Path to save the analysis (default: input_file_analysis.md)",
    )

    args = parser.parse_args()

    # Set default output path if not provided
    if not args.output:
        input_path = Path(args.input_file)
        args.output = str(input_path.with_name(f"{input_path.stem}_analysis.md"))

    # Initialize the parsing controller
    controller = ParsingController()

    # Analyze the text file
    print(f"Analyzing text file: {args.input_file}")
    analysis = controller.analyze_text_file(args.input_file)

    # Save the analysis
    print(f"Saving analysis to: {args.output}")
    controller.save_analysis_to_file(analysis, args.output)

    print("Analysis completed successfully!")


if __name__ == "__main__":
    main()
