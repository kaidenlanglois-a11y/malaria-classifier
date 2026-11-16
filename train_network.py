"""Train the neural network to tell infected cells from healthy ones."""

import random
from pathlib import Path

from data_loader import load_dataset, split_dataset
from model import compute_accuracy, compute_loss
from network import backward, forward, initialise_parameters
from visualise import save_weight_sheet

# How big a step to take downhill each time.
LEARNING_RATE = 0.005

# How many times to go through the whole training set.
EPOCHS = 30

# How many images the network looks at before each update.
BATCH_SIZE = 64

# How many images to load from each class.
IMAGES_PER_CLASS = 5000

# The share of images held back for testing.
TEST_FRACTION = 0.2

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

# Start with small random weights.
parameters = initialise_parameters(train_images.shape[1])

# Keep a copy of the random starting weights, to compare against later.
starting_weights = parameters["hidden_weights"].copy()

rng = random.Random(0)
image_count = len(train_labels)

for epoch in range(EPOCHS):
    # Visit the training images in a fresh random order each epoch.
    order = list(range(image_count))
    rng.shuffle(order)

    # Work through the images one small batch at a time.
    for start in range(0, image_count, BATCH_SIZE):
        batch_rows = order[start : start + BATCH_SIZE]
        batch_images = train_images[batch_rows]
        batch_labels = train_labels[batch_rows]

        # 1. Forward pass on this batch.
        hidden_outputs, predictions = forward(batch_images, parameters)

        # 2. Backward pass: which way is downhill, judged from this batch.
        gradients = backward(
            batch_images, hidden_outputs, predictions, batch_labels, parameters
        )

        # 3. Take a small step downhill.
        for name in parameters:
            parameters[name] = parameters[name] - LEARNING_RATE * gradients[name]

    # After each epoch, measure the loss over the whole training set.
    _, train_predictions = forward(train_images, parameters)
    loss = compute_loss(train_predictions, train_labels)
    print(f"Epoch {epoch}: loss {loss:.4f}")

# Grade the network on the images it trained on, and on images it has never seen.
_, train_predictions = forward(train_images, parameters)
_, test_predictions = forward(test_images, parameters)
print(f"Training accuracy: {compute_accuracy(train_predictions, train_labels):.1%}")
print(f"Test accuracy: {compute_accuracy(test_predictions, test_labels):.1%}")

# Save a picture of what training changed in each hidden neuron's weights.
learned_changes = parameters["hidden_weights"] - starting_weights
save_weight_sheet(learned_changes, Path("hidden_neurons.png"))
print("Saved hidden_neurons.png")