# Load the NumPy library and give it the short nickname "np".
import numpy as np

# Build a 2x2 grid of numbers. Each inner list is one row.
# Think of it as a tiny grayscale image: 0 is black, 255 is white.
pixels = np.array([[0, 128], [255, 64]])

# Show the grid in the terminal.
print(pixels)

# Show the grid's dimensions as (rows, columns). Expect (2, 2).
print(pixels.shape)