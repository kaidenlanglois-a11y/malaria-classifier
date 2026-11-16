"""Train the neuron to tell infected cells from healthy ones."""

import numpy as np

from data_loader import load_dataset
from model import predict

# Load a small sample: 100 images from each class.
images, labels = load_dataset(images_per_class=100)

# Start with every weight at zero: the neuron knows nothing yet.
weights = np.zeros(images.shape[1])
bias = 0.0

# Ask the untrained neuron for its predictions.
predictions = predict(images, weights, bias)
print("First five predictions:", predictions[:5])
print("Last five predictions:", predictions[-5:])