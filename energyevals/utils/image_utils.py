import base64
import json
from typing import Any

from energyevals.agent.schema import ImageContent


def extract_images_from_result(result: str) -> list[ImageContent]:
    """Extract image content from tool result JSON."""
    images = []

    try:
        data = json.loads(result)

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    images.extend(_extract_images_from_dict(item))
        elif isinstance(data, dict):
            images.extend(_extract_images_from_dict(data))

    except (json.JSONDecodeError, TypeError):
        pass

    return images


def _extract_images_from_dict(data: dict[str, Any]) -> list[ImageContent]:
    """Extract images from dictionary recursively."""
    images = []

    if "image_base64" in data:
        images.append(ImageContent(
            image_base64=data["image_base64"],
            media_type=data.get("media_type", "image/jpeg"),
        ))

    if "image_data" in data and "media_type" in data:
        images.append(ImageContent(
            image_base64=data["image_data"],
            media_type=data["media_type"],
        ))

    if "images" in data and isinstance(data["images"], list):
        for img in data["images"]:
            if isinstance(img, dict):
                if "image_base64" in img or "data" in img:
                    b64 = img.get("image_base64") or img.get("data") or ""
                    images.append(ImageContent(
                        image_base64=str(b64),
                        media_type=img.get("media_type", "image/jpeg"),
                    ))

    return images


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def decode_base64_to_bytes(base64_string: str) -> bytes:
    return base64.b64decode(base64_string)
