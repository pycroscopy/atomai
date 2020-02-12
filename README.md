# AtomAI
Module for machine learning based analysis of experimental atom-resolved data.
<br>
<p align="center">
  <img src="https://github.com/ziatdinovmax/atomai/blob/master/AtomAI_logo.png" width="30%" title="AtomAI">
<p align="justify">
<br>

AtomAI has two main modules: *atomnet* and *atomstat*. The *atomnet* allows training a neural network with just one line of code and making a prediction with a trained model (which will take 2 lines of code). The *atomstat* takes the atomnet predictions and performs statistical analysis of the features asssociated with the identified atoms and defects.

Here is an example of how one can train a neural network for atom/defect finding with essentially one line of code:

```python
from atomai import atomnet

# Here you load training data
# ...

# Train a model
trained_model = atomnet.trainer(
    images_all, labels_all, 
    images_test_all, labels_test_all,
    training_cycles=500).run()   
```

The trained model can be used to find atoms/defects in the previously unseen experimental data:
```python
# Here you load new experimental data (as 2D or 3D numpy array)
# ...

# Get raw NN output
nn_input, pred = atomnet.predictor(
    expdata, trained_model).run()
# Transform to atomic classes and coordinates
coordinates = atomnet.locator(pred).run()
```

We can then perform statistical analysis using the network predictions, including finding Gaussian mixture model components, analyzing trajectories of different classes of defects and calculating transition probability between them:
```python
from atomai import atomstat
# Get local descriptors (e.g. subimages centered around impurities)
imstack = atomstat.imlocal(pred, coordinates, r=32, coord_class=1)
# Calculate Gaussian mixture components
components_im, classes_list = imstack.gmm(10, plot_results=True)
# Calculate Gaussian mixture components and the transition frequencies between them
imstack.transition_matrix(10, plot_results=True, plot_values=True)
```

TODO:

0) Test trajectories analysis to atomstat on WS2 data

1) Add several trained models (e.g. graphene, ferroics)

2) Add ferronet-type analysis for a single image to atomstat

3) Add and test class for contour analysis (similar in structure and logic to class 'locator') to atomstat

4) Add comparison between true coordinates and predicted coordinates (histogram and colored overlay)

5) Add more examples and notebooks
