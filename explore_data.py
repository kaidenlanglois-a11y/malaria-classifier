# Path builds file paths that work on any operating system.
from pathlib import Path

import numpy as np
from PIL import Image

# Where the dataset lives, relative to the project folder.
# Capital letters mark a constant: a value that never changes.
DATA_DIR = Path("data") / "cell_images"

# Collect every PNG file in the Parasitized folder, in sorted order.
parasitized_paths = sorted((DATA_DIR / "Parasitized").glob("*.png"))

# How many infected-cell images are there?
print("Parasitized images:", len(parasitized_paths))

# Take the first image path. Counting starts at 0 in Python.
image_path = parasitized_paths[0]

# Open the file with Pillow, then convert it to a NumPy array.
image = np.array(Image.open(image_path))

# Inspect the image as numbers.
print("File:", image_path.name)
print("Shape:", image.shape)
print("Data type:", image.dtype)
print("Smallest value:", image.min())
print("Largest value:", image.max())

# Look at single pixels: image[row, column] gives that pixel's [red, green, blue].
print("Top-left corner pixel:", image[0, 0])
print("Centre pixel:", image[74, 71])