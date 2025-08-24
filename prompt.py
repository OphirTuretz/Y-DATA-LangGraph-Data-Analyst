def read_prompt_file(prompt_file_path: str) -> str:
    """
    Reads a prompt from a specified file.
    Args:
        prompt_file_path (str): The path to the prompt file.
    Returns:
        str: The content of the prompt file.
    """
    with open(prompt_file_path, "r", encoding="utf-8") as f:
        prompt = f.read()
    return prompt
