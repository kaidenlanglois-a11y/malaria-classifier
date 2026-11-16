"""Load malaria cell images and turn them into arrays a model can use."""

from pathlib import Path

import numpy as np
from PIL import Image

# Every image is resized to this many pixels on each side.
IMAGE_SIZE = 32


def load_image(image_path: Path) -> np.ndarray:
    """Load one image as a flat array of numbers between 0 and 1."""
    # Open the file and make sure it has 3 colour channels (red, green, blue).
    image = Image.open(image_path).convert("RGB")

    # Shrink the image so every one has the same dimensions.
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    # Convert to an array of decimal numbers, then scale 0-255 down to 0-1.
    pixels = np.array(image, dtype=np.float64) / 255.0

    # Unroll the 32 x 32 x 3 grid into one long row of 3,072 numbers.
    return pixels.flatten()