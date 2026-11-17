"""Convolution layers with max pooling, built from scratch with NumPy."""

import random

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from network import random_weights, relu

# Each filter looks at a window this many pixels wide and tall.
FILTER_SIZE = 3

# Pooling keeps the largest value in each block this many pixels wide and tall.
POOL_SIZE = 2

# Red, green and blue: the number of values per pixel in the original images.
COLOUR_CHANNELS = 3


def initialise_filters(input_channels: int, filter_count: int, seed: int) -> dict:
    """Create one layer's starting filter weights and biases."""
    rng = random.Random(seed)

    # Each filter has one weight per value in its window:
    # width x height x the number of values at each position.
    weights_per_filter = FILTER_SIZE * FILTER_SIZE * input_channels

    return {
        # One column of weights per filter.
        "filter_weights": random_weights(weights_per_filter, filter_count, rng),
        # One bias per filter.
        "filter_biases": np.zeros(filter_count),
    }


def extract_windows(grids: np.ndarray) -> np.ndarray:
    """Cut out every window of every grid, each as one row of numbers."""
    # Slide a window over the height and width of every grid.
    windows = sliding_window_view(grids, (FILTER_SIZE, FILTER_SIZE), axis=(1, 2))

    # Flatten each window: shape (images, map height, map width, window values).
    return windows.reshape(*windows.shape[:3], -1)


def to_blocks(maps: np.ndarray) -> np.ndarray:
    """Group each map into pooling blocks, dropping any leftover row and column."""
    size = maps.shape[1] // POOL_SIZE
    trimmed = maps[:, : size * POOL_SIZE, : size * POOL_SIZE, :]
    return trimmed.reshape(len(maps), size, POOL_SIZE, size, POOL_SIZE, maps.shape[-1])


def conv_forward(
    grids: np.ndarray, filters: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run grids through one layer: filters, ReLU, then pooling."""
    windows = extract_windows(grids)

    # Every filter takes a weighted sum of every window, then ReLU.
    feature_maps = relu(windows @ filters["filter_weights"] + filters["filter_biases"])

    # Keep the largest value in each pooling block.
    pooled = to_blocks(feature_maps).max(axis=(2, 4))

    # The windows and feature maps are returned too, because training needs them.
    return windows, feature_maps, pooled


def enlarge(grid: np.ndarray) -> np.ndarray:
    """Stretch a pooled grid back up, repeating each value across its block."""
    return np.repeat(np.repeat(grid, POOL_SIZE, axis=1), POOL_SIZE, axis=2)


def conv_backward(
    pooled_errors: np.ndarray,
    grids: np.ndarray,
    windows: np.ndarray,
    feature_maps: np.ndarray,
    pooled: np.ndarray,
    filters: dict,
) -> dict:
    """Work out one layer's gradients, and the blame to pass to the layer before."""
    image_count = len(pooled_errors)
    filter_count = feature_maps.shape[-1]
    map_size = feature_maps.shape[1]

    # The part of each feature map that pooling actually used.
    even = (map_size // POOL_SIZE) * POOL_SIZE
    trimmed = feature_maps[:, :even, :even, :]

    # Stage 1: undo the pooling. Only the position that held the largest value
    # in each block takes the blame. Positions that tied share it equally.
    was_largest = trimmed == enlarge(pooled)
    tie_counts = to_blocks(was_largest).sum(axis=(2, 4))
    map_errors = np.zeros_like(feature_maps)
    map_errors[:, :even, :even, :] = enlarge(pooled_errors / tie_counts) * was_largest

    # Stage 2: undo the ReLU. Positions that output 0 take no blame.
    map_errors = map_errors * (feature_maps > 0)

    # Stage 3: this layer's gradients. Error x input, added up over every
    # position each filter visited.
    flat_windows = windows.reshape(-1, windows.shape[-1])
    flat_errors = map_errors.reshape(-1, filter_count)

    # Stage 4: blame for this layer's inputs. First, each window's share:
    # the map errors times the weights that carried each value forward.
    window_errors = map_errors @ filters["filter_weights"].T
    window_errors = window_errors.reshape(
        image_count, map_size, map_size, grids.shape[-1], FILTER_SIZE, FILTER_SIZE
    )

    # Windows overlap, so each input value sat in several windows. Add up its
    # blame from every window position it appeared in.
    input_errors = np.zeros_like(grids)
    for row in range(FILTER_SIZE):
        for column in range(FILTER_SIZE):
            input_errors[
                :, row : row + map_size, column : column + map_size, :
            ] += window_errors[..., row, column]

    return {
        "filter_weights": flat_windows.T @ flat_errors / image_count,
        "filter_biases": flat_errors.sum(axis=0) / image_count,
        "input_errors": input_errors,
    }