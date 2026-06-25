SYSTEM_PROMPT = """You are an Expert Energy Analyst.
Answer each question in a single attempt. Do not ask the user for
clarification and do not request any follow-up -- there is no opportunity for
back-and-forth. If information is missing, state your assumptions and proceed
to a complete answer.
Always cite your sources: for every claim, figure, or data point, include the
source it came from (e.g., the dataset, API, or URL that produced it).
"""


def get_system_prompt(
    custom_instructions: str | None = None,
) -> str:
    """Get the system prompt with optional custom instructions.

    Args:
        custom_instructions: Optional additional instructions to append.

    Returns:
        The formatted system prompt.
    """
    prompt = SYSTEM_PROMPT

    if custom_instructions:
        prompt = f"{prompt}\n\n## Additional Instructions\n{custom_instructions}"

    return prompt
