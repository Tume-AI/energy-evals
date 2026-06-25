from energyevals.utils import extract_images_from_result


class ImageProcessor:
    """Processor for extracting images from tool results.

    Extracts base64-encoded images from tool results and converts them
    to a format suitable for inclusion in messages.
    """

    def extract_images(self, result: str) -> list[dict[str, str]]:
        """Extract base64 images from tool result.

        Args:
            result: Tool result as JSON string

        Returns:
            List of image dictionaries with 'base64' and 'media_type' keys
        """
        images_content = extract_images_from_result(result)

        return [
            {
                "base64": img.image_base64,
                "media_type": img.media_type,
            }
            for img in images_content
            if img.image_base64
        ]
