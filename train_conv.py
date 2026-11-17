"""Train the convolutional network to tell infected cells from healthy ones."""

import random
from pathlib import Path

import numpy as np

from conv import (
    COLOUR_CHANNELS,
    FILTER_SIZE,
    POOL_SIZE,
    conv_backward,
    conv_forward,
    initialise_filters,
)
from data_loader import IMAGE_SIZE, load_dataset, split_dataset
from model import compute_accuracy, compute_loss
from network import backward, forward, initialise_parameters
from visualise import save_filter_sheet

# The step size at the beginning of training.
LEARNING_RATE = 0.003

# Each epoch, the step size shrinks to this fraction of the one before.
LEARNING_RATE_DECAY = 0.95

# How much of the previous step's direction each update keeps.
MOMENTUM = 0.9

# How many times to go through the whole training set.
EPOCHS = 20

# How many images the network looks at before each update.
BATCH_SIZE = 64

# How many filters each convolution layer has.
FIRST_FILTER_COUNT = 8
SECOND_FILTER_COUNT = 16

# How many images to load from each class.
IMAGES_PER_CLASS = 5000

# The share of all images held back for the final test.
TEST_FRACTION = 0.2

# The share of the remaining images held back for comparing experiments.
VALIDATION_FRACTION = 0.2

# Leave this False while experimenting. Set it to True only for the final,
# one-time measurement, after every setting has been chosen.
RUN_FINAL_TEST = False

# How many images to push through at once when only predicting.
PREDICT_BATCH_SIZE = 500


def pooled_size(input_size: int) -> int:
    """Return the width of a layer's pooled output, given the width of its input."""
    return (input_size - FILTER_SIZE + 1) // POOL_SIZE


def to_grids(images: np.ndarray) -> np.ndarray:
    """Fold flat images back into grids of colour pixels."""
    return images.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, COLOUR_CHANNELS)


def forward_pass(
    grids: np.ndarray, first_filters: dict, second_filters: dict, parameters: dict
) -> dict:
    """Run images through both convolution layers and then the dense network."""
    # Layer 1 looks at the image. Layer 2 looks at layer 1's pooled feature maps.
    first_windows, first_maps, first_pooled = conv_forward(grids, first_filters)
    second_windows, second_maps, second_pooled = conv_forward(
        first_pooled, second_filters
    )

    # Flatten layer 2's output into one row of features per image.
    features = second_pooled.reshape(len(second_pooled), -1)
    hidden_outputs, predictions = forward(features, parameters)

    # Keep every intermediate result, because the backward pass needs them.
    return {
        "first_windows": first_windows,
        "first_maps": first_maps,
        "first_pooled": first_pooled,
        "second_windows": second_windows,
        "second_maps": second_maps,
        "second_pooled": second_pooled,
        "features": features,
        "hidden_outputs": hidden_outputs,
        "predictions": predictions,
    }


def predict_all(
    grids: np.ndarray, first_filters: dict, second_filters: dict, parameters: dict
) -> np.ndarray:
    """Predict for many images, a batch at a time to keep memory use low."""
    predictions = []

    for start in range(0, len(grids), PREDICT_BATCH_SIZE):
        batch_grids = grids[start : start + PREDICT_BATCH_SIZE]
        steps = forward_pass(batch_grids, first_filters, second_filters, parameters)
        predictions.append(steps["predictions"])

    # Join the batches back into one long array.
    return np.concatenate(predictions)


def step_downhill(
    weights: dict, gradients: dict, velocities: dict, learning_rate: float
) -> None:
    """Update one set of weights, carrying momentum over from earlier steps."""
    for name in weights:
        # Keep most of the previous step's direction, then add the new push.
        velocities[name] = MOMENTUM * velocities[name] - learning_rate * gradients[name]

        # Move the weights by the current velocity.
        weights[name] = weights[name] + velocities[name]


# Load a random sample and set aside the test set.
print("Loading images...")
images, labels = load_dataset(IMAGES_PER_CLASS)
train_images, train_labels, test_images, test_labels = split_dataset(
    images, labels, TEST_FRACTION
)

# Split what is left into a training set and a validation set.
train_images, train_labels, validation_images, validation_labels = split_dataset(
    train_images, train_labels, VALIDATION_FRACTION, seed=1
)
print("Training images:", len(train_labels))
print("Validation images:", len(validation_labels))
print("Test images:", len(test_labels))

# Centre the inputs: find each pixel's average and the overall spread,
# using the training images only.
pixel_means = train_images.mean(axis=0)
pixel_spread = train_images.std()

# Shift and scale all three sets with the training set's numbers, then fold
# each flat image back into a grid for the convolution layers.
train_grids = to_grids((train_images - pixel_means) / pixel_spread)
validation_grids = to_grids((validation_images - pixel_means) / pixel_spread)
test_grids = to_grids((test_images - pixel_means) / pixel_spread)

# Work out how many features the second layer hands to the dense network.
second_pooled_size = pooled_size(pooled_size(IMAGE_SIZE))
feature_count = second_pooled_size * second_pooled_size * SECOND_FILTER_COUNT

# Start with small random weights everywhere. Layer 1 reads 3 colour values
# per position. Layer 2 reads one value per first-layer filter.
first_filters = initialise_filters(COLOUR_CHANNELS, FIRST_FILTER_COUNT, seed=1)
second_filters = initialise_filters(FIRST_FILTER_COUNT, SECOND_FILTER_COUNT, seed=2)
parameters = initialise_parameters(feature_count)

# Every weight starts with no velocity.
parameter_velocities = {name: 0.0 for name in parameters}
second_velocities = {name: 0.0 for name in second_filters}
first_velocities = {name: 0.0 for name in first_filters}

rng = random.Random(0)
image_count = len(train_labels)

for epoch in range(EPOCHS):
    # Take smaller steps as training goes on, to settle into the valley.
    learning_rate = LEARNING_RATE * LEARNING_RATE_DECAY**epoch

    # Visit the training images in a fresh random order each epoch.
    order = list(range(image_count))
    rng.shuffle(order)

    # Work through the images one small batch at a time.
    for start in range(0, image_count, BATCH_SIZE):
        batch_rows = order[start : start + BATCH_SIZE]
        batch_grids = train_grids[batch_rows]
        batch_labels = train_labels[batch_rows]

        # 1. Forward pass: image -> layer 1 -> layer 2 -> dense network.
        steps = forward_pass(batch_grids, first_filters, second_filters, parameters)

        # 2. Backward pass, in the opposite order. Each stage hands its
        #    "input_errors" to the stage before it.
        gradients = backward(
            steps["features"],
            steps["hidden_outputs"],
            steps["predictions"],
            batch_labels,
            parameters,
        )
        second_gradients = conv_backward(
            gradients["input_errors"].reshape(steps["second_pooled"].shape),
            steps["first_pooled"],
            steps["second_windows"],
            steps["second_maps"],
            steps["second_pooled"],
            second_filters,
        )
        first_gradients = conv_backward(
            second_gradients["input_errors"],
            batch_grids,
            steps["first_windows"],
            steps["first_maps"],
            steps["first_pooled"],
            first_filters,
        )

        # 3. Take a step downhill in all three parts, with momentum.
        step_downhill(parameters, gradients, parameter_velocities, learning_rate)
        step_downhill(
            second_filters, second_gradients, second_velocities, learning_rate
        )
        step_downhill(first_filters, first_gradients, first_velocities, learning_rate)

    # After each epoch, check progress on the training and validation sets.
    train_predictions = predict_all(
        train_grids, first_filters, second_filters, parameters
    )
    validation_predictions = predict_all(
        validation_grids, first_filters, second_filters, parameters
    )
    loss = compute_loss(train_predictions, train_labels)
    validation_accuracy = compute_accuracy(validation_predictions, validation_labels)
    print(
        f"Epoch {epoch}: loss {loss:.4f}, "
        f"validation accuracy {validation_accuracy:.1%} "
        f"(step size {learning_rate:.5f})"
    )

# Final scores. The validation score is the one to compare between experiments.
train_accuracy = compute_accuracy(train_predictions, train_labels)
print(f"Training accuracy: {train_accuracy:.1%}")
print(f"Validation accuracy: {validation_accuracy:.1%}")

# The test set is only touched when explicitly asked for.
if RUN_FINAL_TEST:
    test_predictions = predict_all(
        test_grids, first_filters, second_filters, parameters
    )
    print(f"Test accuracy: {compute_accuracy(test_predictions, test_labels):.1%}")

# Save everything needed to use the trained network later without retraining:
# all the weights, plus the numbers used to centre the inputs.
np.savez(
    "model_weights.npz",
    first_filter_weights=first_filters["filter_weights"],
    first_filter_biases=first_filters["filter_biases"],
    second_filter_weights=second_filters["filter_weights"],
    second_filter_biases=second_filters["filter_biases"],
    **parameters,
    pixel_means=pixel_means,
    pixel_spread=pixel_spread,
)
print("Saved model_weights.npz")

# Save a picture of the small pattern each first-layer filter responds to.
save_filter_sheet(first_filters["filter_weights"], Path("conv_filters.png"))
print("Saved conv_filters.png")