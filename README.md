[![PyPI version](https://badge.fury.io/py/atomai.svg)](https://badge.fury.io/py/atomai)
[![Build Status](https://travis-ci.com/ziatdinovmax/atomai.svg?branch=master)](https://travis-ci.com/ziatdinovmax/atomai)
[![Documentation Status](https://readthedocs.org/projects/atomai/badge/?version=latest)](https://atomai.readthedocs.io/en/latest/?badge=latest)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/8fa8829627f040dda46e2dc30e48aca1)](https://app.codacy.com/manual/ziatdinovmax/atomai?utm_source=github.com&utm_medium=referral&utm_content=ziatdinovmax/atomai&utm_campaign=Badge_Grade_Dashboard)
[![Downloads](https://pepy.tech/badge/atomai/month)](https://pepy.tech/project/atomai/month)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ziatdinovmax/atomai/blob/master/examples/notebooks/Quickstart_AtomAI_in_the_Cloud.ipynb)
[![Gitpod ready-to-code](https://img.shields.io/badge/Gitpod-ready--to--code-blue?logo=gitpod)](https://gitpod.io/#https://github.com/ziatdinovmax/atomai)

# AtomAI

## What is AtomAI?

AtomAI is a simple Python package for machine learning-based analysis of experimental atom-resolved data from electron and scanning probe microscopes, which doesn't require any advanced knowledge of Python (or machine learning). It is the next iteration of the [AICrystallographer project](https://github.com/pycroscopy/AICrystallographer).

## How to use it?

AtomAI has two main modules: *atomnet* and *atomstat*. The *atomnet* is for training neural networks (with just one line of code) and for applying trained models to finding atoms and defects in image data (which also takes  a single line of code). The *atomstat* allows taking the *atomnet* predictions and performing the statistical analysis on the local image descriptors associated with the identified atoms and defects (e.g., principal component analysis of atomic distortions in a single image or computing gaussian mixture model components with the transition probabilities for movies).

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

# Get model's "raw" prediction, atomic coordinates and classes
nn_input, (nn_output, coordinates) = atomnet.predictor(expdata, trained_model, refine=False).run()
```

One can then perform statistical analysis using the information extracted by *atomnet*. For example, for a single image, one can identify domains with different ferroic distortions:

```python
from atomai import atomstat

# Get local descriptors
imstack = atomstat.imlocal(nn_output, coordinates, crop_size=32, coord_class=1)

# Compute distortion "eigenvectors" with associated loading maps and plot results:
nmf_results = imstack.imblock_nmf(n_components=4, plot_results=True)
```

For movies, one can extract trajectories of individual defects and calculate the transition probabilities between different classes:

```python
# Get local descriptors (such as subimages centered around impurities)
imstack = atomstat.imlocal(nn_output, coordinates, crop_size=32, coord_class=1)

# Calculate Gaussian mixture model (GMM) components
components, imgs, coords = imstack.gmm(n_components=10, plot_results=True)

# Calculate GMM components and transition probabilities for different trajectories
traj_all, trans_all, fram_all = imstack.transition_matrix(n_components=10, rmax=10)

# and more
```

## Quickstart: AtomAI in the Cloud

The easiest way to start using AtomAI is via [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb) 

1) [Use AtomAI to train a deep NN for atom finding](https://colab.research.google.com/github/ziatdinovmax/atomai/blob/master/examples/notebooks/atomai_atomnet.ipynb)

2) [Analyze distortion domains in a single atomic image](https://colab.research.google.com/github/ziatdinovmax/atomai/blob/master/examples/notebooks/atomai_atomstat.ipynb)

3) Analyze trajectories of atomic defects in atomic movie - TBA

4) [Prepare training data from experimental image with atomic coordinates (beta)](https://colab.research.google.com/github/ziatdinovmax/atomai/blob/master/examples/notebooks/atomai_training_data.ipynb)

5) [Atom finding using deep ensembles for uncertainty quantification (advanced)](https://colab.research.google.com/github/ziatdinovmax/atomai/blob/master/examples/notebooks/atomai_deep_ensembles.ipynb)

## Installation
First, install [PyTorch](https://pytorch.org/get-started/locally/). Then, install AtomAI via

```bash
pip install atomai
```
