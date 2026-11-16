"""Train the neural network to tell infected cells from healthy ones."""

from data_loader import load_dataset, split_dataset
from model import compute_accuracy, compute_loss
from network import backward, forward, initialise_parameters

# How big a step to take downhill each time.
LEARNING_RATE = 0.005

# How many times to go through the whole training set.
EPOCHS = 1000

# How many images to load from each class.
IMAGES_PER_CLASS = 1000

# The share of images held back for testing.
TEST_FRACTION = 0.2

# Load a random sample, then set aside a test set the network never trains on.
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

for epoch in range(EPOCHS):
    # 1. Forward pass: run the images through the network.
    hidden_outputs, predictions = forward(train_images, parameters)

    # 2. Backward pass: work out which way is downhill for every weight.
    gradients = backward(
        train_images, hidden_outputs, predictions, train_labels, parameters
    )

    # 3. Take a small step downhill for each set of weights and biases.
    for name in parameters:
        parameters[name] = parameters[name] - LEARNING_RATE * gradients[name]

    # Report progress every 100 epochs.
    if epoch % 100 == 0:
        loss = compute_loss(predictions, train_labels)
        print(f"Epoch {epoch}: loss {loss:.4f}")

# Grade the network on the images it trained on, and on images it has never seen.
_, train_predictions = forward(train_images, parameters)
_, test_predictions = forward(test_images, parameters)
print(f"Training accuracy: {compute_accuracy(train_predictions, train_labels):.1%}")
print(f"Test accuracy: {compute_accuracy(test_predictions, test_labels):.1%}")