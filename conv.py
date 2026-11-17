"""A convolution layer with max pooling, built from scratch with NumPy."""

import random

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from data_loader import IMAGE_SIZE
from network import random_weights, relu

# Each filter looks at a window this many pixels wide and tall.
FILTER_SIZE = 3

# How many filters the layer has. Each learns its own small pattern.
FILTER_COUNT = 8

# Pooling keeps the largest value in each block this many pixels wide and tall.
POOL_SIZE = 2

# Red, green and blue.
COLOUR_CHANNELS = 3

# How many positions a window fits across the image: 
MAP_SIZE = IMAGE_SIZE - FILTER_SIZE + 1

# Pooling halves the width and height of each feature map.
POOLED_SIZE = MAP_SIZE // POOL_SIZE

# The number of values the layer hands on per image
FEATURE_COUNT = POOLED_SIZE * POOLED_SIZE * FILTER_COUNT


def initialise_filters(seed: int = 1) -> dict:
    """Create the filters' starting weights and biases."""
    rng = random.Random(seed)

    # Each filter has one weight per value in its window:
    # width x height X colours .
    weights_per_filter = FILTER_SIZE * FILTER_SIZE * COLOUR_CHANNELS

    return {
        # One column of  weights per filter: shape (weights_per_filter, FILTER_COUNT).
        "filter_weights": random_weights(weights_per_filter, FILTER_COUNT, rng),
        # One bias per filter.
        "filter_biases": np.zeros(FILTER_COUNT),
    }


def extract_windows(images: np.ndarray) -> np.ndarray:
    """Cut out every window of every image, each as a row of  numbers."""
    # Fold each flat image back into a 32 x 32 grid of colour pixels.
    grids = images.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, COLOUR_CHANNELS)

    # Slide a 3 x 3 window over the height and width of every image.
    windows = sliding_window_view(grids, (FILTER_SIZE, FILTER_SIZE), axis=(1, 2))

    # Flatten each window, giving shape (images, MAP_SIZE, MAP_SIZE, FILTER_COUNT).
    return windows.reshape(len(grids), MAP_SIZE, MAP_SIZE, -1)


def conv_forward(
    images: np.ndarray, filters: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run images through the filters, ReLU and pooling."""
    windows = extract_windows(images)

    # Every filter takes a weighted sum of every window, then ReLU.
    # Shape: (images, 30, 30, FILTER_COUNT).
    feature_maps = relu(windows @ filters["filter_weights"] + filters["filter_biases"])

    # Group each feature map into 2 x 2 blocks and keep the largest value in each.
    blocks = feature_maps.reshape(
        -1, POOLED_SIZE, POOL_SIZE, POOLED_SIZE, POOL_SIZE, FILTER_COUNT
    )
    pooled = blocks.max(axis=(2, 4))

    # Flatten the pooled maps into one row of features per image.
    features = pooled.reshape(len(pooled), -1)

    # The windows and feature maps are returned too, because training will need them.
    return windows, feature_maps, features


def enlarge(grid: np.ndarray) -> np.ndarray:
    """Stretch a pooled grid back to feature-map size, repeating each value 2 x 2."""
    return np.repeat(np.repeat(grid, POOL_SIZE, axis=1), POOL_SIZE, axis=2)


def conv_backward(
    feature_errors: np.ndarray,
    windows: np.ndarray,
    feature_maps: np.ndarray,
    features: np.ndarray,
) -> dict:
    """Work out the gradient for every filter weight and bias."""
    image_count = len(feature_errors)

    # Fold the flat features and their errors back into pooled-map shape.
    pooled_shape = (-1, POOLED_SIZE, POOLED_SIZE, FILTER_COUNT)
    pooled = features.reshape(pooled_shape)
    pooled_errors = feature_errors.reshape(pooled_shape)

    # Stage 1: undo the pooling. Only the position that held the largest value
    # in each 2 x 2 block was passed forward, so only it takes the blame.
    was_largest = feature_maps == enlarge(pooled)

    # If several positions tied for largest, they share the blame equally.
    tie_blocks = was_largest.reshape(
        -1, POOLED_SIZE, POOL_SIZE, POOLED_SIZE, POOL_SIZE, FILTER_COUNT
    )
    tie_counts = tie_blocks.sum(axis=(2, 4))
    map_errors = enlarge(pooled_errors / tie_counts) * was_largest

    # Stage 2: undo the ReLU. Positions that output 0 take no blame.
    map_errors = map_errors * (feature_maps > 0)

    # Stage 3: error x input, added up over every position each filter visited.
    flat_windows = windows.reshape(-1, windows.shape[-1])
    flat_errors = map_errors.reshape(-1, FILTER_COUNT)

    return {
        "filter_weights": flat_windows.T @ flat_errors / image_count,
        "filter_biases": flat_errors.sum(axis=0) / image_count,
    }