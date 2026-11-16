"""Turn the network's learned weights into pictures."""

import math
from pathlib import Path

import numpy as np
from PIL import Image

from data_loader import IMAGE_SIZE

# Each small weight picture is enlarged to this many pixels per side.
TILE_SIZE = 128

# How many pictures sit in each row of the sheet.
TILES_PER_ROW = 4


def weights_to_tile(neuron_weights: np.ndarray) -> Image.Image:
    """Turn one hidden neuron's weights into a small colour picture."""
    # Arrange the 3,072 weights back into image shape: 32 x 32 pixels x 3 colours.
    grid = neuron_weights.reshape(IMAGE_SIZE, IMAGE_SIZE, 3)

    # Scale so a weight of zero becomes mid-grey, the largest positive weight
    # becomes bright, and the largest negative weight becomes dark.
    largest = np.abs(grid).max()
    pixels = (grid / largest * 0.5 + 0.5) * 255

    # Enlarge without blurring, so each weight shows as a crisp square.
    tile = Image.fromarray(pixels.astype(np.uint8))
    return tile.resize((TILE_SIZE, TILE_SIZE), Image.Resampling.NEAREST)


def save_weight_sheet(hidden_weights: np.ndarray, output_path: Path) -> None:
    """Save one picture per hidden neuron, laid out in a grid."""
    neuron_count = hidden_weights.shape[1]
    row_count = math.ceil(neuron_count / TILES_PER_ROW)

    # A blank sheet big enough to hold every tile.
    sheet = Image.new("RGB", (TILES_PER_ROW * TILE_SIZE, row_count * TILE_SIZE))

    for neuron in range(neuron_count):
        # Column `neuron` of the grid holds that neuron's weights.
        tile = weights_to_tile(hidden_weights[:, neuron])

        # Work out where this tile goes on the sheet.
        row = neuron // TILES_PER_ROW
        column = neuron % TILES_PER_ROW
        sheet.paste(tile, (column * TILE_SIZE, row * TILE_SIZE))

    sheet.save(output_path)