"""Load malaria cell images and turn them into arrays a model can use."""

from pathlib import Path

import numpy as np
from PIL import Image

# Every image is resized to this many pixels on each side.
IMAGE_SIZE = 32

# Where the dataset lives, relative to the project folder.
DATA_DIR = Path("data") / "cell_images"

# Each class folder and the number the model uses for it.
LABELS = {"Uninfected": 0, "Parasitized": 1}


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


def load_dataset(images_per_class: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Load a random sample of images from both classes, with a label for each."""
    # A seeded generator makes the same "random" choices on every run.
    rng = np.random.default_rng(seed)
    images = []
    labels = []

    # Go through each class folder in turn.
    for class_name, label in LABELS.items():
        image_paths = sorted((DATA_DIR / class_name).glob("*.png"))

        # Pick positions at random from the whole folder, with no repeats.
        chosen = rng.choice(len(image_paths), size=images_per_class, replace=False)

        for index in chosen:
            images.append(load_image(image_paths[index]))
            labels.append(label)

    # Turn the lists into arrays: one row per image.
    return np.array(images), np.array(labels)


def split_dataset(
    images: np.ndarray, labels: np.ndarray, test_fraction: float, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle the dataset and split it into a training set and a test set."""
    rng = np.random.default_rng(seed)

    # The row numbers 0, 1, 2, ... in a random order.
    order = rng.permutation(len(labels))

    # The first slice of shuffled rows becomes the test set; the rest is for training.
    test_size = int(len(labels) * test_fraction)
    test_rows = order[:test_size]
    train_rows = order[test_size:]

    return images[train_rows], labels[train_rows], images[test_rows], labels[test_rows]