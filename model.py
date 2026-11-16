"""A single artificial neuron, built from scratch with NumPy."""

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Squash any number into the range 0 to 1."""
    return 1.0 / (1.0 + np.exp(-z))


def predict(images: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    """Return the neuron's confidence (0 to 1) that each image is infected."""
    # Weighted sum: multiply each pixel by its weight, add them up, add the bias.
    z = images @ weights + bias

    # Squash each sum into a confidence between 0 and 1.
    return sigmoid(z)
