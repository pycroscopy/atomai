# AtomAI
**Under active development (i.e. breaking changes can occur on a daily basis)**

<p align="center">
  <img src="https://github.com/ziatdinovmax/atomai/blob/master/AtomAI_logo-v2.png" width="30%" title="AtomAI">
<p align="justify">

## What is AtomAI?

AtomAI is a simple Python package for machine learning based analysis of experimental atom-resolved data from electron and scanning probe microscopes, which doesn't require any advanced knowledge of Python (or machine learning).

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

To perform statistical analysis for the identified atoms and defects:
```python
from atomai import atomstat

# Get local descriptors (such as subimages centered around impurities)
imstack = atomstat.imlocal(nn_output, coordinates, r=32, coord_class=1)

# Calculate Gaussian mixture components (GMM)
components_im, classes_list = imstack.gmm(n_components=10, plot_results=True)

# For movies, calculate GMM and the transition probabilities between them along the trajectories
trans_all, traj_all, fram_all = imstack.transition_matrix(n_components=10, rmax=10)

# and more
```

## Quickstart: AtomAI in the Cloud

1. [Use AtomAI to train a deep NN for atom finding in Colab](https://colab.research.google.com/github/ziatdinovmax/atomai/blob/master/notebooks/atomai_atomnet.ipynb)
2. Analyze trajectories of atomic defects in atomic movie - TBA
3. TBA

## Local Installation

TBA

## TODO

1) Add ferronet-type analysis for a single image to atomstat

2) Add and test class for contour analysis to atomstat

3) Add comparison between "true" coordinates and predicted coordinates to utils as a measure of model performance

4) Add several trained models (e.g. graphene, ferroics)

5) Add more examples and notebooks
