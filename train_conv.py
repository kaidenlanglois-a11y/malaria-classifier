"""Train the convolutional network to tell infected cells from healthy ones."""

import random

import numpy as np

from conv import FEATURE_COUNT, conv_backward, conv_forward, initialise_filters
from data_loader import load_dataset, split_dataset
from model import compute_accuracy, compute_loss
from network import backward, forward, initialise_parameters

# step size at the begining.
LEARNING_RATE = 0.01    

# Each epoch, the step size shrinks to this fraction of the one before
LEARNING_RATE_DECAY = 0.95

# How many times to go through the whole training set.
EPOCHS = 30

# How many images the network looks at before each update.
BATCH_SIZE = 64

# How many images to load from each class.
IMAGES_PER_CLASS = 5000

# The share of images held back for testing.
TEST_FRACTION = 0.2

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


# Load a random sample, then set aside a test set the network never trains on.
print("Loading images...")
images, labels = load_dataset(IMAGES_PER_CLASS)
train_images, train_labels, test_images, test_labels = split_dataset(
    images, labels, TEST_FRACTION
)
print("Training images:", len(train_labels))
print("Test images:", len(test_labels))

# Centre the inputs: find each pixel's average and the overall spread,
# using the training images only.
pixel_means = train_images.mean(axis=0)
pixel_spread = train_images.std()

# Shift and scale both sets with the training set's numbers.
train_images = (train_images - pixel_means) / pixel_spread
test_images = (test_images - pixel_means) / pixel_spread

# Start with small random weights for both halves. The back half now takes
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

    # After each epoch, measure the loss over the whole training set.
    train_predictions = predict_all(train_images, filters, parameters)
    loss = compute_loss(train_predictions, train_labels)
    print(f"Epoch {epoch}: loss {loss:.4f} (step size {learning_rate:.5f})")

# Grade the network on the images it trained on, and on images it has never seen.
train_predictions = predict_all(train_images, filters, parameters)
test_predictions = predict_all(test_images, filters, parameters)
print(f"Training accuracy: {compute_accuracy(train_predictions, train_labels):.1%}")
print(f"Test accuracy: {compute_accuracy(test_predictions, test_labels):.1%}")