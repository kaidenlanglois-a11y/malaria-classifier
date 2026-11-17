"""Train the convolutional network to tell infected cells from healthy ones."""

import random

import numpy as np

from conv import FEATURE_COUNT, conv_backward, conv_forward, initialise_filters
from data_loader import load_dataset, split_dataset
from model import compute_accuracy, compute_loss
from network import backward, forward, initialise_parameters

# The step size at the beginning of training.
LEARNING_RATE = 0.01

# Each epoch, the step size shrinks to this fraction of the one before.
LEARNING_RATE_DECAY = 0.95

# How many times to go through the whole training set.
EPOCHS = 30

# How many images the network looks at before each update.
BATCH_SIZE = 64

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


def predict_all(images: np.ndarray, filters: dict, parameters: dict) -> np.ndarray:
    """Predict for many images, a batch at a time to keep memory use low."""
    predictions = []

    for start in range(0, len(images), PREDICT_BATCH_SIZE):
        batch_images = images[start : start + PREDICT_BATCH_SIZE]

        # Front half: filters turn the images into features.
        _, _, features = conv_forward(batch_images, filters)

        # Back half: the network turns the features into predictions.
        _, batch_predictions = forward(features, parameters)
        predictions.append(batch_predictions)

    # Join the batches back into one long array.
    return np.concatenate(predictions)


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

# Shift and scale all three sets with the training set's numbers.
train_images = (train_images - pixel_means) / pixel_spread
validation_images = (validation_images - pixel_means) / pixel_spread
test_images = (test_images - pixel_means) / pixel_spread

# Start with small random weights for both halves. The back half takes
# the filters' features as its input, not raw pixels.
filters = initialise_filters()
parameters = initialise_parameters(FEATURE_COUNT)

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
        batch_images = train_images[batch_rows]
        batch_labels = train_labels[batch_rows]

        # 1. Forward pass: images -> filters -> features -> prediction.
        windows, feature_maps, features = conv_forward(batch_images, filters)
        hidden_outputs, predictions = forward(features, parameters)

        # 2. Backward pass: the network first, then the filters, using the
        #    blame the network passed back to its inputs.
        gradients = backward(
            features, hidden_outputs, predictions, batch_labels, parameters
        )
        filter_gradients = conv_backward(
            gradients["input_errors"], windows, feature_maps, features
        )

        # 3. Take a small step downhill in both halves.
        for name in parameters:
            parameters[name] = parameters[name] - learning_rate * gradients[name]
        for name in filters:
            filters[name] = filters[name] - learning_rate * filter_gradients[name]

    # After each epoch, check progress on the training and validation sets.
    train_predictions = predict_all(train_images, filters, parameters)
    validation_predictions = predict_all(validation_images, filters, parameters)
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
    test_predictions = predict_all(test_images, filters, parameters)
    print(f"Test accuracy: {compute_accuracy(test_predictions, test_labels):.1%}")