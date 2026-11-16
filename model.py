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


def compute_loss(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Measure how wrong the predictions are. Lower is better; 0 is perfect."""
    # Keep predictions slightly away from exactly 0 or 1, because log(0) is undefined.
    predictions = np.clip(predictions, 1e-12, 1 - 1e-12)

    # Part that counts for infected cells: a big penalty when the prediction is near 0.
    infected_part = labels * np.log(predictions)

    # Part that counts for healthy cells: a big penalty when the prediction is near 1.
    healthy_part = (1 - labels) * np.log(1 - predictions)

    # Average over all images, and flip the sign so the loss is positive.
    return -np.mean(infected_part + healthy_part)


def compute_gradients(
    images: np.ndarray, predictions: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, float]:
    """Work out how to change each weight and the bias to reduce the loss."""
    # How far off each prediction is: positive means the guess was too high.
    errors = predictions - labels

    # For each weight: the average of (error x that pixel's value) over all images.
    weight_gradients = images.T @ errors / len(labels)

    # For the bias: the average error.
    bias_gradient = np.mean(errors)

    return weight_gradients, bias_gradient


def compute_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Return the fraction of images the neuron labelled correctly."""
    # Count a prediction as "infected" when confidence is at least 0.5.
    return np.mean((predictions >= 0.5) == labels)