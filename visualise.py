"""Turn the network's learned weights into pictures."""

import math
from pathlib import Path

import numpy as np
from PIL import Image

from conv import COLOUR_CHANNELS, FILTER_SIZE
from data_loader import IMAGE_SIZE

# Each small weight picture is enlarged to this many pixels per side.
TILE_SIZE = 128

# The width of the grey border between pictures, in pixels.
TILE_GAP = 8

# How many pictures sit in each row of the sheet.
TILES_PER_ROW = 4


def grid_to_tile(grid: np.ndarray) -> Image.Image:
    """Turn a grid of weights (height x width x colours) into an enlarged picture."""
    # Scale so a weight of zero becomes mid-grey, the largest positive weight
    # becomes bright, and the largest negative weight becomes dark.
    largest = np.abs(grid).max()
    pixels = (grid / largest * 0.5 + 0.5) * 255

    # Enlarge without blurring, so each weight shows as a crisp square.
    tile = Image.fromarray(pixels.astype(np.uint8))
    return tile.resize((TILE_SIZE, TILE_SIZE), Image.Resampling.NEAREST)


def save_tile_sheet(grids: list[np.ndarray], output_path: Path) -> None:
    """Save one picture per grid, laid out in rows on a single sheet."""
    row_count = math.ceil(len(grids) / TILES_PER_ROW)

    # The distance from the start of one tile to the start of the next.
    stride = TILE_SIZE + TILE_GAP

    # A grey sheet big enough to hold every tile, with a border all round.
    sheet_size = (TILES_PER_ROW * stride + TILE_GAP, row_count * stride + TILE_GAP)
    sheet = Image.new("RGB", sheet_size, (128, 128, 128))

    for index, grid in enumerate(grids):
        # Work out where this tile goes on the sheet.
        row = index // TILES_PER_ROW
        column = index % TILES_PER_ROW
        position = (TILE_GAP + column * stride, TILE_GAP + row * stride)
        sheet.paste(grid_to_tile(grid), position)

    sheet.save(output_path)


def save_weight_sheet(hidden_weights: np.ndarray, output_path: Path) -> None:
    """Save a picture of each hidden neuron's weights over the whole image."""
    # Column `neuron` holds that neuron's weights. Fold each back into image shape.
    grids = [
        hidden_weights[:, neuron].reshape(IMAGE_SIZE, IMAGE_SIZE, COLOUR_CHANNELS)
        for neuron in range(hidden_weights.shape[1])
    ]
    save_tile_sheet(grids, output_path)


def save_filter_sheet(filter_weights: np.ndarray, output_path: Path) -> None:
    """Save a picture of each convolution filter's small window of weights."""
    # A window's values are stored colour by colour, so fold each filter into
    # (colours, height, width), then reorder to (height, width, colours).
    grids = [
        filter_weights[:, index]
        .reshape(COLOUR_CHANNELS, FILTER_SIZE, FILTER_SIZE)
        .transpose(1, 2, 0)
        for index in range(filter_weights.shape[1])
    ]
    save_tile_sheet(grids, output_path)