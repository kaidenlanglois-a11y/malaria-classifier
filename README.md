# Malaria Cell Classifier

A neural network that tells malaria-infected red blood cells from healthy ones,
built from scratch in NumPy with no machine learning library. I wrote the forward
pass, the loss, backpropagation and the training loop by hand as a way to learn
how neural networks work.

## Results

Trained on 8,000 cell images and tested on 2,000 the network never saw.

| Model | Test accuracy |
| --- | --- |
| Single neuron (logistic regression) | 62.7% |
| One hidden layer, raw pixels | 60.0% |
| + centred inputs | 66.0% |
| + 10,000 images | 70.5% |
| + mini-batch training | 71.7% |

## What the network learned

Each tile shows how training changed one hidden neuron's weights. The network
found the round cell region by itself and learned to respond to the purple stain
of the parasite.

![What each hidden neuron learned](hidden_neurons.png)

Every neuron is a soft blob over the whole cell, not a small dot detector. That
is the limit of a fully connected network on this task, and the reason the next
step is convolution.

## How it works

- `data_loader.py` loads the images, resizes them to 32 x 32, and scales the pixels.
- `model.py` holds a single neuron: sigmoid, loss, gradients and accuracy.
- `network.py` holds the network: parameters, forward pass and backpropagation.
- `train.py` trains the single neuron.
- `train_network.py` trains the network with mini-batch gradient descent.
- `visualise.py` turns the learned weights into a picture.

## Running it

1. Download the cell images from the
   [NLM malaria data page](https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-datasheet.html)
   (the link named NLM-Falciparum-Thin-Cell-Images) and extract them so the
   images sit in `data/cell_images/Parasitized` and `data/cell_images/Uninfected`.
2. Create a virtual environment and install the packages:

   ```
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. Train the network:

   ```
   python train_network.py
   ```

## Data

The images come from the US National Library of Medicine's malaria dataset:
27,558 single-cell images from thin blood smears, half parasitized and half
uninfected.

## Next steps

- Add convolutional layers so the network can find a parasite anywhere in the cell.
- Train on the full dataset.