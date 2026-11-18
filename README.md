# Malaria Cell Classifier

A neural network that tells malaria-infected red blood cells from healthy ones,
built from scratch in NumPy with no machine learning library. I wrote the forward
pass, the loss, backpropagation, convolution, pooling, momentum and the training
loop by hand as a way to learn how neural networks work.

**Final result: 94.9% accuracy on 4,800 cell images the model never saw.**

## Results

Each model is scored on cell images it never saw during training.

| Model | Accuracy on unseen images |
| --- | --- |
| Single neuron (logistic regression) | 62.7% |
| One hidden layer, raw pixels | 60.0% |
| + centred inputs | 66.0% |
| + 10,000 images | 70.5% |
| + mini-batch training | 71.7% |
| Convolution layer with max pooling | 79.4% |
| + 24,000 images | 83.8% |
| Second convolution layer (back on the smaller image set) | 79.4% |
| + momentum | 87.7% |
| + 24,000 images | 94.9% |
| **Final score on the held-back test set** | **94.9%** |

The first five rows were measured on a test set that was also used to compare
experiments, so they are slightly optimistic. From the convolution rows on, the
scores are validation accuracy: a separate slice of the data used to choose
settings such as the learning rate.

The last row is the honest one. The test set of 4,800 images was held back the
whole time, played no part in any decision, and was scored exactly once, after
every setting had been fixed. It matched the validation score, which means the
model was not over-tuned to the validation images.

Two things stood out along the way. A second convolution layer gave no gain at
all with plain gradient descent; it only paid off once momentum was added. And
more data helped every time it was tried.

## What the networks learned

### Dense network

Each tile shows how training changed one hidden neuron's weights. The network
found the round cell region by itself and learned to respond to the purple stain
of the parasite.

![What each hidden neuron learned](hidden_neurons.png)

Every neuron is a soft blob over the whole cell, not a small dot detector. A
dense network needs separate weights for every position, so it cannot learn
"a dot anywhere". That is the limit convolution removes.

### Convolutional network

Each tile is one filter from the first layer: a 3 x 3 window of weights that
slides across the whole image. The same purple-against-green contrast appears,
now as small local patterns that can be detected at any position in the cell.
The second layer combines these small patterns into larger ones.

![The eight first-layer convolution filters](conv_filters.png)

## How it works

- `data_loader.py` loads the images, resizes them to 32 x 32, and scales the pixels.
- `model.py` holds a single neuron: sigmoid, loss, gradients and accuracy.
- `network.py` holds the dense network: parameters, forward pass and backpropagation.
- `conv.py` holds the convolution layer: filters, max pooling, and their backpropagation.
- `train.py` trains the single neuron.
- `train_network.py` trains the dense network with mini-batch gradient descent.
- `train_conv.py` trains the final model: two convolution layers and a dense
  layer, with momentum, learning rate decay, and separate training, validation
  and test sets (15,360 / 3,840 / 4,800 images).
- `visualise.py` turns learned weights into pictures.

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

3. Train the convolutional network:

   ```
   python train_conv.py
   ```

   To score the held-back test set as well, set `RUN_FINAL_TEST = True` at the
   top of `train_conv.py`. It is off by default so the test set is not used
   while experimenting.

## Data

The images come from the US National Library of Medicine's malaria dataset:
27,558 single-cell images from thin blood smears, half parasitized and half
uninfected.

## Next steps

- Draw feature maps to show where each filter fires on a cell.
- Look at the cells the model gets wrong to see what they have in common.