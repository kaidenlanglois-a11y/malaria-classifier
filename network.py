"""A small neural network with one hidden layer, built from scratch with NumPy."""

import random

import numpy as np

from model import sigmoid

# How many neurons sit in the hidden layer.
HIDDEN_SIZE = 16


def random_weights(rows: int, columns: int, rng: random.Random) -> np.ndarray:
    """Make a grid of small random numbers, sized for a layer with `rows` inputs."""
    # Layers with more inputs get smaller starting weights, so sums stay modest.
    scale = (2.0 / rows) ** 0.5

    # Draw each number from a bell curve centred on zero.
    return np.array(
        [[rng.gauss(0.0, scale) for _ in range(columns)] for _ in range(rows)]
    )


def initialise_parameters(input_size: int, seed: int = 0) -> dict:
    """Create the network's starting weights and biases."""
    rng = random.Random(seed)

    return {
        # One column of weights per hidden neuron: shape (input_size, HIDDEN_SIZE).
        "hidden_weights": random_weights(input_size, HIDDEN_SIZE, rng),
        # One bias per hidden neuron, starting at zero.
        "hidden_biases": np.zeros(HIDDEN_SIZE),
        # One weight per hidden neuron, feeding the output neuron: shape (HIDDEN_SIZE,).
        "output_weights": random_weights(HIDDEN_SIZE, 1, rng).flatten(),
        # The output neuron's single bias.
        "output_bias": 0.0,
    }


def relu(z: np.ndarray) -> np.ndarray:
    """Turn negative numbers into 0 and leave positive numbers unchanged."""
    return np.maximum(0.0, z)


def forward(images: np.ndarray, parameters: dict) -> tuple[np.ndarray, np.ndarray]:
    """Run images through the network: pixels -> hidden layer -> prediction."""
    # Hidden layer: 16 weighted sums per image, one per hidden neuron.
    hidden_sums = images @ parameters["hidden_weights"] + parameters["hidden_biases"]

    # Squash the hidden sums with ReLU.
    hidden_outputs = relu(hidden_sums)

    # Output neuron: a weighted sum of the 16 hidden outputs.
    output_sum = hidden_outputs @ parameters["output_weights"] + parameters["output_bias"]

    # Squash into a confidence between 0 and 1.
    predictions = sigmoid(output_sum)

    # The hidden outputs are returned too, because training will need them.
    return hidden_outputs, predictions

def backward(
    images: np.ndarray,
    hidden_outputs: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    parameters: dict,
) -> dict:
    """Work out the gradient for every weight and bias (backpropagation)."""
    image_count = len(labels)

    # Stage 1: how far off each final prediction is.
    errors = predictions - labels

    # Gradients for the output neuron: error x its inputs (the hidden outputs).
    output_weight_gradients = hidden_outputs.T @ errors / image_count
    output_bias_gradient = np.mean(errors)

    # Stage 2: pass the blame back. Each hidden neuron's share of an image's
    # error is that error times the output weight that listened to the neuron.
    hidden_errors = np.outer(errors, parameters["output_weights"])

    # Neurons that were switched off by ReLU (output of 0) take no blame.
    hidden_errors = hidden_errors * (hidden_outputs > 0)

    # Stage 3: gradients for the hidden layer: hidden error x its inputs (the pixels).
    hidden_weight_gradients = images.T @ hidden_errors / image_count
    hidden_bias_gradients = np.mean(hidden_errors, axis=0)

    # Use the same names as the parameters, so each gradient matches its weight.
    return {
        "hidden_weights": hidden_weight_gradients,
        "hidden_biases": hidden_bias_gradients,
        "output_weights": output_weight_gradients,
        "output_bias": output_bias_gradient,
    }