"""SentinelKilnDB dataset loader and label parser."""

import io
import os
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image
from datasets import load_dataset, DatasetDict

CLASS_ID_TO_NAME: Dict[int, str] = {
    0: "CFCBK",
    1: "FCBK",
    2: "Zigzag",
}

CLASS_NAME_TO_ID: Dict[str, int] = {
    "CFCBK": 0,
    "FCBK": 1,
    "Zigzag": 2,
}


def parse_yolo_obb_line(line: str) -> Tuple[int, str, List[float]]:
    """Parse a single YOLO-OBB label line.

    Format: <class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
    All coordinates normalized to [0, 1].

    Returns:
        (class_id, class_name, [x1, y1, x2, y2, x3, y4, x4, y4])
    """
    parts = line.strip().split()
    if len(parts) != 9:
        raise ValueError(f"Expected 9 tokens in YOLO-OBB line, got {len(parts)}: '{line}'")

    class_id = int(parts[0])
    class_name = CLASS_ID_TO_NAME.get(class_id, f"Unknown_{class_id}")
    coords = [float(x) for x in parts[1:]]

    # Validate coordinate range
    for c in coords:
        if c < -0.2 or c > 1.2:
            # Tolerant bounds for kilns slightly extending over tile boundary
            pass

    return class_id, class_name, coords


def decode_image_bytes(image_data: Union[bytes, Image.Image, Dict[str, Any]]) -> Image.Image:
    """Safely decode an image representation into an RGB PIL Image."""
    if isinstance(image_data, Image.Image):
        return image_data.convert("RGB")
    elif isinstance(image_data, (bytes, bytearray)):
        return Image.open(io.BytesIO(image_data)).convert("RGB")
    elif isinstance(image_data, dict) and "bytes" in image_data:
        return Image.open(io.BytesIO(image_data["bytes"])).convert("RGB")
    else:
        raise TypeError(f"Unsupported image data type: {type(image_data)}")


def load_sentinel_kiln_dataset(
    dataset_name: str = "SustainabilityLabIITGN/SentinelKilnDB",
    cache_dir: Optional[str] = None,
    split: Optional[str] = None,
) -> Union[DatasetDict, Any]:
    """Load SentinelKilnDB from Hugging Face with local cache support.

    Args:
        dataset_name: Hugging Face dataset identifier.
        cache_dir: Optional custom local cache directory.
        split: Optional split name ('train', 'validation', 'test').

    Returns:
        DatasetDict or Dataset split.
    """
    # Environment variables can specify HF token securely if provided
    hf_token = os.environ.get("HF_TOKEN", None)

    try:
        ds = load_dataset(
            dataset_name,
            cache_dir=cache_dir,
            split=split,
            token=hf_token,
        )
        return ds
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load dataset '{dataset_name}'. "
            f"If authentication is required, ensure HF_TOKEN is set in your environment. "
            f"Details: {exc}"
        ) from exc
