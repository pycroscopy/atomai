# AtomAI
**Under active development (i.e. breaking changes can occur on a daily basis)**

## What is AtomAI?

AtomAI is a simple Python package for machine learning based analysis of experimental atom-resolved data from electron and scanning probe microscopes, which doesn't require any advanced knowledge of Python (or machine learning).

## How to use it?

AtomAI has two main modules: *atomnet* and *atomstat*. The *atomnet* is for training neural networks (with just one line of code) and for applying trained models to finding atoms and defects in image data (which takes two lines of code). The *atomstat* allows taking the *atomnet* predictions and performing the statistical analysis (e.g., Gaussian mixture modelling, transition probability calculations) on the local image descriptors corresponding to the identified atoms and defects.

Here is an example of how one can train a neural network for atom/defect finding with essentially one line of code:

```python
from atomai import atomnet

# Here you load your training data
dataset = np.load('training_data.npz')
images_all = dataset['X_train']
labels_all = dataset['y_train']
images_test_all = dataset['X_test']
labels_test_all = dataset['y_test']

# Train a model
trained_model = atomnet.trainer(
    images_all, labels_all, 
    images_test_all, labels_test_all,
    training_cycles=500).run()   
```

Trained models can be used to find atoms/defects in the previously unseen (by a model) experimental data:
```python
# Here you load new experimental data (as 2D or 3D numpy array)
expdata = np.load('expdata-test.npy')

# Get raw NN output
nn_input, nn_output = atomnet.predictor(expdata, trained_model).run()
    
# Transform to atomic classes and coordinates
coordinates = atomnet.locator(nn_output).run()
```

One can then perform statistical analysis using the information extracted by *atomnet*. For example, for a single image, one can identify domains with different ferroic distortions:

```python
from atomai import atomstat

# Get local descriptors
imstack = atomstat.imlocal(nn_output, coordinates, r=32, coord_class=1)

# Compute distortion "eigenvectors" with associated loading maps and plot results:
nmf_results = imstack.imblock_nmf(4, plot_results=True)
```

For movies, one can extract trajectories of individual defects and calculate the transition probabilities between different classes:

```python
# Get local descriptors (such as subimages centered around impurities)
imstack = atomstat.imlocal(nn_output, coordinates, r=32, coord_class=1)

# Calculate GMM components and transition probabilities for different trajectories
trans_all, traj_all, fram_all = imstack.transition_matrix(n_components=10, rmax=10)

# and more
```

## Quickstart: AtomAI in the Cloud

1. [Use AtomAI to train a deep NN for atom finding in Colab](https://colab.research.google.com/github/ziatdinovmax/atomai/blob/master/examples/notebooks/atomai_atomnet.ipynb)
2. [Analyze distortion domains in a single atomic image](https://colab.research.google.com/github/ziatdinovmax/atomai/blob/master/examples/notebooks/atomai_atomstat.ipynb)
3. Analyze trajectories of atomic defects in atomic movie - TBA

## Installation
Install AtomAI using

```pip install -q git+https://github.com/ziatdinovmax/atomai.git```


## TODO

1) Add test modules

2) Add comparison between "true" coordinates and predicted coordinates as a measure of model performance

3) Add class for the analysis of particles geometry

4) Add more examples and notebooks

5) Add docs
