from pathlib import Path

from energyevals.agent.constants import CSV_THRESHOLD

from energyevals.agent.processors.csv_processor import CSVProcessor
from energyevals.agent.processors.image_processor import ImageProcessor


class ResultProcessor:
    """Composite processor for tool results.

    Combines CSV processing and image extraction into a single processor.
    """

    def __init__(
        self,
        csv_threshold: int = CSV_THRESHOLD,
        csv_output_dir: str | Path = "./agent_outputs",
    ):
        """Initialize result processor.

        Args:
            csv_threshold: Row count threshold for saving to CSV
            csv_output_dir: Directory to save CSV files
        """
        self.csv_processor = CSVProcessor(
            threshold=csv_threshold,
            output_dir=csv_output_dir,
        )
        self.image_processor = ImageProcessor()

    def process_result(
        self,
        tool_name: str,
        result: str,
    ) -> tuple[str, str | None]:
        """Process tool result for CSV saving.

        Args:
            tool_name: Name of the tool that produced the result
            result: Tool result as JSON string

        Returns:
            Tuple of (context_result, csv_path)
        """
        return self.csv_processor.process(tool_name, result)

    def extract_images(self, result: str) -> list[dict[str, str]]:
        """Extract images from tool result.

        Args:
            result: Tool result as JSON string

        Returns:
            List of image dictionaries
        """
        return self.image_processor.extract_images(result)
