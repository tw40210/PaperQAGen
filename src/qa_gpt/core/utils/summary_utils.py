from pydantic import BaseModel


def get_total_summary_fields(summaries: dict[str, BaseModel]) -> int:
    """Calculate the total number of fields across all summary model_dumps.

    Args:
        summaries (Dict[str, BaseModel]): Dictionary of summary objects

    Returns:
        int: Total number of fields across all summary model_dumps
    """
    total_fields = 0
    for summary in summaries.values():
        # Count fields from the actual model_dump
        total_fields += len(summary.model_dump())
    return total_fields
