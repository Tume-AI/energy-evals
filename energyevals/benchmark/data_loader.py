import csv
from pathlib import Path

from energyevals.benchmark.models import Question


_REQUIRED_COLUMNS = {"S/N", "Category", "Question type", "Difficulty level", "Question"}


def load_questions(csv_path: Path) -> list[Question]:
    """Load benchmark questions from a CSV file.

    Expected columns: ``S/N``, ``Category``, ``Question type``,
    ``Difficulty level``, ``Question``.

    Args:
        csv_path: Path to the CSV file containing questions.

    Returns:
        List of Question objects.

    Raises:
        ValueError: If required columns are missing or a row has an invalid S/N value.
    """
    questions = []

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file is empty or has no header: {csv_path}")
        missing = _REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"CSV file {csv_path} is missing required columns: {sorted(missing)}. "
                f"Expected: {sorted(_REQUIRED_COLUMNS)}"
            )
        for row_num, row in enumerate(reader, start=2):  # row 1 is the header
            raw_id = row.get("S/N", "").strip()
            try:
                question_id = int(raw_id)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Invalid S/N value '{raw_id}' at row {row_num} in {csv_path}. "
                    f"Expected an integer."
                )
            questions.append(
                Question(
                    id=question_id,
                    category=row.get("Category", ""),
                    question_type=row.get("Question type", ""),
                    difficulty=row.get("Difficulty level", ""),
                    question=row.get("Question", ""),
                )
            )

    return questions
