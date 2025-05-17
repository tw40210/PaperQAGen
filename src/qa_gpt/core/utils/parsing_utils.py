def extract_question_set_id(file_name: str) -> str:
    """Extract question set ID from a question file name.

    Args:
        file_name: The name of the question file (e.g. 'mc_question_StandardSummary_conclusion_0.json')

    Returns:
        str: The question set ID (e.g. 'StandardSummary_conclusion_0')

    Examples:
        >>> extract_question_set_id('mc_question_StandardSummary_conclusion_0.json')
        'StandardSummary_conclusion_0'
        >>> extract_question_set_id('mc_question_0.json')
        '0'
    """
    # Remove the file extension
    name_without_ext = file_name.rsplit(".", 1)[0]

    # Remove the 'mc_question_' prefix
    if name_without_ext.startswith("mc_question_"):
        return name_without_ext[len("mc_question_") :]

    return name_without_ext
