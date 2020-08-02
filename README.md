[![PyPI version](https://badge.fury.io/py/atomai.svg)](https://badge.fury.io/py/atomai)
[![Build Status](https://travis-ci.com/ziatdinovmax/atomai.svg?branch=master)](https://travis-ci.com/ziatdinovmax/atomai)
[![Documentation Status](https://readthedocs.org/projects/atomai/badge/?version=latest)](https://atomai.readthedocs.io/en/latest/?badge=latest)

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/8fa8829627f040dda46e2dc30e48aca1)](https://app.codacy.com/manual/ziatdinovmax/atomai?utm_source=github.com&utm_medium=referral&utm_content=ziatdinovmax/atomai&utm_campaign=Badge_Grade_Dashboard)
[![Downloads](https://pepy.tech/badge/atomai/month)](https://pepy.tech/project/atomai/month)

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ziatdinovmax/atomai/blob/master/examples/notebooks/Quickstart_AtomAI_in_the_Cloud.ipynb)
[![Gitpod ready-to-code](https://img.shields.io/badge/Gitpod-ready--to--code-blue?logo=gitpod)](https://gitpod.io/#https://github.com/ziatdinovmax/atomai)

# AtomAI

## What is AtomAI

AtomAI is a simple Python package for machine learning-based analysis of experimental atom-resolved data from electron and scanning probe microscopes, which doesn't require any advanced knowledge of Python (or machine learning). It is the next iteration of the [AICrystallographer project](https://github.com/pycroscopy/AICrystallographer).

## How to use it

AtomAI has two main modules: *atomnet* and *atomstat*. The *atomnet* is for training neural networks (with just one line of code) and for applying trained models to finding atoms and defects in image data (which also takes  a single line of code). The *atomstat* allows taking the *atomnet* predictions and performing the statistical analysis on the local image descriptors associated with the identified atoms and defects (e.g., principal component analysis of atomic distortions in a single image or computing gaussian mixture model components with the transition probabilities for movies).

### Quickstart: AtomAI in the Cloud

The easiest way to start using AtomAI is via [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb) 

1) [Train a deep fully convolutional neural network for atom finding](https://colab.research.google.com/github/ziatdinovmax/atomai/blob/master/examples/notebooks/atomai_atomnet.ipynb)

2) [Multivariate statistical analysis of distortion domains in a single atomic image](https://colab.research.google.com/github/ziatdinovmax/atomai/blob/master/examples/notebooks/atomai_atomstat.ipynb)

3) [Variational autoencoders for analysis of structural transformations](https://colab.research.google.com/github/ziatdinovmax/atomai/blob/master/examples/notebooks/atomai_vae.ipynb)

4) [Prepare training data from experimental image with atomic coordinates](https://colab.research.google.com/github/ziatdinovmax/atomai/blob/master/examples/notebooks/atomai_training_data.ipynb)

### Model training
Below is an example of how one can train a neural network for atom/defect finding with essentially one line of code:

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

One can also train an ensemble of models instead of just a single model. The average ensemble prediction is usually more accurate and reliable than that of the single model. In addition, we also get the information about the [uncertainty in our prediction](https://arxiv.org/abs/1612.01474) for each pixel. Note that in the example below, we "augmnent" our data on-the-fly by applying various image transformations (e.g. adding Gaussian noise, changing contrast, rotating and zooming). The sequence of these transformations is different for every model in the ensemble ensuring a unique "training trajectory" for each model.

```python
# Initialize ensemble trainer
trainer = atomnet.ensemble_trainer(X_train, y_train, n_models=12,
                                   rotation=True, zoom=True, contrast=True, 
                                   gauss=True, blur=True, background=True, 
                                   loss='ce', batch_size=16, training_cycles_base=1000,
                                   training_cycles_ensemble=100, filename='ensemble')
# train deep ensemble of models
basemodel, ensemble, ensemble_aver = trainer.run()
```

### Prediction with trained model(s)

Trained models are be used to find atoms/particles/defects in the previously unseen (by a model) experimental data:

```python
# Here we load new experimental data (as 2D or 3D numpy array)
expdata = np.load('expdata-test.npy')

# Get model's "raw" prediction, atomic coordinates and classes
nn_input, (nn_output, coord_class) = atomnet.predictor(trained_model, refine=False).run(expdata)

# Get ensemble prediction (mean and variance for "raw" prediction and coordinates)
epredictor = atomnet.ensemble_predictor(basemodel, ensemble, calculate_coordinates=True)
(out_mu, out_var), (coord_mu, coord_var) = epredictor.run(expdata)
```

(Note: The deep ensemble-based prediction of atomic coordinates mean and variance uses DBSCAN method to arrange predictions from different ensemble models into clusters and the result is sensitive to the value of *eps* passed as ```**kwargs``` (default is 0.5). Sometimes it is better to simply run atomnet.locator (thresholding followed by finding blob centers) on the mean "raw" prediction (out_mu) of the ensemble)

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
