# AtomAI
**Under active development (i.e. breaking changes can occur on a daily basis)**

<p align="center">
  <img src="https://github.com/ziatdinovmax/atomai/blob/master/AtomAI_logo.png" width="30%" title="AtomAI">
<p align="justify">

## What is AtomAI?

AtomAI is a simple Python package for machine learning based analysis of experimental atom-resolved data from electron and scanning probe microscopes, which doesn't require any advanced knowledge of Python (or machine learning).

AtomAI has two main modules: *atomnet* and *atomstat*. The *atomnet* is for training neural networks (with just one line of code) and for applying trained models to finding atoms and defects in image data (which takes two lines of code). The *atomstat* allows taking the *atomnet* predictions and performing the statistical analysis (e.g., Gaussian mixture modelling, transition probability calculations) on the local image descriptors corresponding to the identified atoms and defects.

Here is an example of how one can train a neural network for atom/defect finding with essentially one line of code:

```python
from atomai import atomnet

# Here you load your training data
# ...

# Train a model
trained_model = atomnet.trainer(
    images_all, labels_all, 
    images_test_all, labels_test_all,
    training_cycles=500).run()   
```

Trained models can be used to find atoms/defects in the previously unseen (by a model) experimental data:
```python
# Here you load new experimental data (as 2D or 3D numpy array)
# ...

# Get raw NN output
nn_input, pred = atomnet.predictor(expdata, trained_model).run()
    
# Transform to atomic classes and coordinates
coordinates = atomnet.locator(pred).run()
```

To perform statistical analysis for the identified atoms and defects:
```python
from atomai import atomstat

# Get local descriptors (such as subimages centered around impurities)
imstack = atomstat.imlocal(pred, coordinates, r=32, coord_class=1)

# Calculate Gaussian mixture components
components_im, classes_list = imstack.gmm(n_components=10, plot_results=True)

# For movies, calculate Gaussian mixture components and the transition probabilities between them
imstack.transition_matrix(n_components=10, plot_results=True, plot_values=True)

# and more
```

## Quickstart: AtomAI in the Cloud

## Local Installation

TBA

**TODO:**

1) Test atomstat functions for analysis of trajectories on WS2 data

2) Add ferronet-type analysis for a single image to atomstat

3) Add and test class for contour analysis to atomstat

4) Add comparison between "true" coordinates and predicted coordinates to utils as a measure of model performance

5) Add several trained models (e.g. graphene, ferroics)

6) Add examples and notebooks
