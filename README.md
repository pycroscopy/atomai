[![PyPI version](https://badge.fury.io/py/atomai.svg)](https://badge.fury.io/py/atomai)
[![Build Status](https://travis-ci.com/pycroscopy/atomai.svg?branch=master)](https://travis-ci.com/pycroscopy/atomai)
[![Documentation Status](https://readthedocs.org/projects/atomai/badge/?version=latest)](https://atomai.readthedocs.io/en/latest/?badge=latest)

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/8fa8829627f040dda46e2dc30e48aca1)](https://app.codacy.com/manual/ziatdinovmax/atomai?utm_source=github.com&utm_medium=referral&utm_content=ziatdinovmax/atomai&utm_campaign=Badge_Grade_Dashboard)
[![Downloads](https://pepy.tech/badge/atomai/month)](https://pepy.tech/project/atomai/month)

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pycroscopy/atomai/blob/master/examples/notebooks/Quickstart_AtomAI_in_the_Cloud.ipynb)
[![Gitpod ready-to-code](https://img.shields.io/badge/Gitpod-ready--to--code-blue?logo=gitpod)](https://gitpod.io/#https://github.com/pycroscopy/atomai)

# AtomAI

## What is AtomAI

AtomAI is a simple Python package for machine learning-based analysis of experimental atomic-scale and mesoscale data from electron and scanning probe microscopes, which doesn't require any advanced knowledge of Python (or machine learning). It is the next iteration of the [AICrystallographer project](https://github.com/pycroscopy/AICrystallographer).

## How to use it

AtomAI has two main modules: *atomnet* and *atomstat*. The *atomnet* is for training neural networks (with just one line of code) and for applying trained models to finding atoms and defects in image data. The *atomstat* allows taking the *atomnet* predictions and performing the statistical analysis on the local image descriptors associated with the identified atoms and defects (e.g., principal component analysis of atomic distortions in a single image or computing gaussian mixture model components with the transition probabilities for movies).

### Quickstart: AtomAI in the Cloud

The easiest way to start using AtomAI is via [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb) 

1) [Train a deep fully convolutional neural network for atom finding](https://colab.research.google.com/github/pycroscopy/atomai/blob/master/examples/notebooks/atomai_atomnet.ipynb)

2) [Multivariate statistical analysis of distortion domains in a single atomic image](https://colab.research.google.com/github/pycroscopy/atomai/blob/master/examples/notebooks/atomai_atomstat.ipynb)

3) [Variational autoencoders for analysis of structural transformations](https://colab.research.google.com/github/pycroscopy/atomai/blob/master/examples/notebooks/atomai_vae.ipynb)

4) [Prepare training data from experimental image with atomic coordinates](https://colab.research.google.com/github/pycroscopy/atomai/blob/master/examples/notebooks/atomai_training_data.ipynb)

### Model training
Below is an example of how one can train a neural network for atom/particle/defect finding with essentially one line of code:

```python
from atomai import atomnet

# Load your training/test data (as numpy arrays or lists of numpy arrays)
dataset = np.load('training_data.npz')
images_all, labels_all, images_test_all, labels_test_all = dataset.values()

# Train a model
trained_model = atomnet.train_single_model(
    images_all, labels_all, images_test_all, labels_test_all,  # train and test data
    gauss_noise=True, zoom=True,  # on-the-fly data augmentation
    training_cycles=500, swa=True)  # train for 500 iterations with stochastic weights averaging at the end  
```

One can also train an ensemble of models instead of just a single model. The average ensemble prediction is usually more accurate and reliable than that of the single model. In addition, we also get the information about the [uncertainty in our prediction](https://arxiv.org/abs/1612.01474) for each pixel.

```python
# Initialize ensemble trainer
etrainer = atomnet.ensemble_trainer(
    images_all, labels_all, images_test_all, labels_test_all,
    rotation=True, zoom=True, gauss_noise=True, # On-the fly data augmentation
    strategy="from_baseline", swa=True, n_models=30, model="dilUnet",
    training_cycles_base=1000, training_cycles_ensemble=100)
    
# Train deep ensemble of models
ensemble, amodel = etrainer.run()
```

### Prediction with trained model(s)

Trained model is used to find atoms/particles/defects in the previously unseen (by a model) experimental data:

```python
# Here we load new experimental data (as 2D or 3D numpy array)
expdata = np.load('expdata.npy')

# Initialize predictive object (can be reused for other datasets)
spredictor = atomnet.predictor(trained_model, use_gpu=True, refine=False)
# Get model's "raw" prediction, atomic coordinates and classes
nn_output, coord_class = spredictor.run(expdata)
```

One can also make a prediction with uncertainty estimates using the ensemble of models:
```python
epredictor = atomnet.ensemble_predictor(amodel, ensemble, calculate_coordinates=True, eps=0.5)
(out_mu, out_var), (coord_mu, coord_var) = epredictor.run(expdata)
```

(Note: In some cases, it may be easier to get coordinates by simply running ```atomnet.locator(*args, *kwargs).run(out_mu)``` on the mean "raw" prediction of the ensemble)

### Statistical analysis

The information extracted by *atomnet* can be further used for statistical analysis of raw and "decoded" data. For example, for a single atom-resolved image of ferroelectric material, one can identify domains with different ferroic distortions:

```python
from atomai import atomstat

# Get local descriptors
imstack = atomstat.imlocal(nn_output, coordinates, window_size=32, coord_class=1)

# Compute distortion "eigenvectors" with associated loading maps and plot results:
pca_results = imstack.imblock_pca(n_components=4, plot_results=True)
```

For movies, one can extract trajectories of individual defects and calculate the transition probabilities between different classes:

```python
# Get local descriptors (such as subimages centered around impurities)
imstack = atomstat.imlocal(nn_output, coordinates, window_size=32, coord_class=1)

# Calculate Gaussian mixture model (GMM) components
components, imgs, coords = imstack.gmm(n_components=10, plot_results=True)

# Calculate GMM components and transition probabilities for different trajectories
transitions_dict = imstack.transition_matrix(n_components=10, rmax=10)

# and more
```
### Variational autoencoders

In addition to multivariate statistical analysis, one can also use [variational autoencoders (VAEs)](https://arxiv.org/abs/1906.02691) in AtomAI to find in the unsupervised fashion the most effective reduced representation of system's local descriptors. The VAEs can be applied to both raw data and NN output, but typically work better with the latter.
```python
from atomai import atomstat, utils

# Get stack of subimages from a movie
imstack, com, frames = utils.extract_subimages(decoded_imgs, coords, window_size=32)

# Initialize and train rotationally-invariant VAE
rvae = atomstat.rVAE(imstack, latent_dim=2, training_cycles=200)
rvae.run()

# Visualize the learned manifold
rvae.manifold2d()
```

## Installation
First, install [PyTorch](https://pytorch.org/get-started/locally/). Then, install AtomAI via

```bash
pip install atomai
```
