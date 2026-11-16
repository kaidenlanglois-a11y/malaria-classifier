"""Train the neuron to tell infected cells from healthy ones."""

import numpy as np

from data_loader import load_dataset, split_dataset
from model import compute_accuracy, compute_gradients, compute_loss, predict

# How big a step to take downhill each time.
LEARNING_RATE = 0.002

# How many times to go through the whole training set.
EPOCHS = 500

# How many images to load from each class.
IMAGES_PER_CLASS = 1000

# The share of images held back for testing.
TEST_FRACTION = 0.2

# Load a random sample, then set aside a test set the neuron never trains on.
images, labels = load_dataset(IMAGES_PER_CLASS)
train_images, train_labels, test_images, test_labels = split_dataset(
    images, labels, TEST_FRACTION
)
print("Training images:", len(train_labels))
print("Test images:", len(test_labels))

# Start with every weight at zero: the neuron knows nothing yet.
weights = np.zeros(train_images.shape[1])
bias = 0.0

for epoch in range(EPOCHS):
    # 1. Ask the neuron for its current predictions.
    predictions = predict(train_images, weights, bias)

    # 2. Work out which way is downhill.
    weight_gradients, bias_gradient = compute_gradients(
        train_images, predictions, train_labels
    )

    # 3. Take a small step in that direction.
    weights = weights - LEARNING_RATE * weight_gradients
    bias = bias - LEARNING_RATE * bias_gradient

    # Report progress every 50 epochs.
    if epoch % 50 == 0:
        loss = compute_loss(predictions, train_labels)
        print(f"Epoch {epoch}: loss {loss:.4f}")

# Grade the neuron on the images it trained on, and on images it has never seen.
train_accuracy = compute_accuracy(predict(train_images, weights, bias), train_labels)
test_accuracy = compute_accuracy(predict(test_images, weights, bias), test_labels)
print(f"Training accuracy: {train_accuracy:.1%}")
print(f"Test accuracy: {test_accuracy:.1%}")