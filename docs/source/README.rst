ReadMe
======

.. image:: https://badge.fury.io/py/atomai.svg
        :target: https://badge.fury.io/py/atomai
        :alt: PyPI version
.. image:: https://travis-ci.com/pycroscopy/atomai.svg?branch=master
        :target: https://travis-ci.com/pycroscopy/atomai
        :alt: Build Status
.. image:: https://readthedocs.org/projects/atomai/badge/?version=latest
        :target: https://atomai.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status
.. image:: https://api.codacy.com/project/badge/Grade/8fa8829627f040dda46e2dc30e48aca1
        :target: https://app.codacy.com/manual/ziatdinovmax/atomai?utm_source=github.com&utm_medium=referral&utm_content=ziatdinovmax/atomai&utm_campaign=Badge_Grade_Dashboard
        :alt: Codacy Badge
.. image:: https://pepy.tech/badge/atomai/month
        :target: https://pepy.tech/project/atomai/month
        :alt: Downloads


.. image:: https://colab.research.google.com/assets/colab-badge.svg
        :target: https://colab.research.google.com/github/pycroscopy/atomai/blob/master/examples/notebooks/Quickstart_AtomAI_in_the_Cloud.ipynb
        :alt: Colab
.. image:: https://img.shields.io/badge/Gitpod-ready--to--code-blue?logo=gitpod
        :target: https://gitpod.io/#https://github.com/pycroscopy/atomai
        :alt: Gitpod ready-to-code

What is AtomAI
--------------
AtomAI is a Pytorch-based package for deep/machine learning analysis of microscopy data, which doesn't require any advanced knowledge of Python (or machine learning). It is the next iteration of the `AICrystallographer project <https://github.com/pycroscopy/AICrystallographer>`_. The intended audience is domain scientists with basic knowledge of how to use NumPy and Matplotlib.

How to use it
-------------

Quickstart: AtomAI in the Cloud
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The easiest way to start using AtomAI is via `Google Colab <https://colab.research.google.com/notebooks/intro.ipynb>`_

1) `Train a Deep Fully Convolutional Neural Network for Atom Finding <https://colab.research.google.com/github/pycroscopy/atomai/blob/master/examples/notebooks/AtomicSemanticSegmention.ipynb>`_

2) `Im2Spec: Predicting 1D Spectral Data from 2D Image Data <https://colab.research.google.com/github/pycroscopy/atomai/blob/master/examples/notebooks/ImSpec.ipynb>`_

3) `Variational Autoencoders I: Learning Disentangled Representations of Arbitrarily Rotated Handwritten Digits <https://colab.research.google.com/github/pycroscopy/atomai/blob/master/examples/notebooks/atomai_rVAE_digits.ipynb>`_

4) `Variational Autoencoders II: Simple Analysis of Structural Transformations in Atomic Movies <https://colab.research.google.com/github/pycroscopy/atomai/blob/master/examples/notebooks/rVAE_graphene.ipynb>`_

5) `Implementation of Custom Image Denoiser in AtomAI <https://colab.research.google.com/github/pycroscopy/atomai/blob/master/examples/notebooks/atomai_custom_model.ipynb>`_

6) `Prepare Training Data From Experimental Image with Atomic Coordinates <https://colab.research.google.com/github/pycroscopy/atomai/blob/master/examples/notebooks/atomai_training_data.ipynb>`_

Semantic segmentation
^^^^^^^^^^^^^^^^^^^^^^

If your goal is to train and/or apply deep learning models for semantic segmentation of your experimental images, it is recommended to start with ```atomai.models.Segmentor```, which provides an easy way to train neural networks (with just two lines of code) and to make a prediction with trained models (with just one line of code). Here is an example of how one can train a neural network for atom/particle/defect finding with essentially two lines of code:

>>> import atomai as aoi
>>> # Initialize model
>>> model = aoi.models.Segmentor(nb_classes=3)  # uses UNet by default
>>> # Train
>>> model.fit(images, labels, images_test, labels_test, # training data (numpy arrays)
>>>           training_cycles=300, compute_accuracy=True, swa=True) # training parameters

Here ```swa``` stands for `stochastic weight averaging <https://arxiv.org/abs/1803.05407>`_,  which usually allows improving the model's accuracy and leads to better generalization. The trained model can be used to find atoms/particles/defects in new, previously unseen (by a model) data:

>>> nn_output, coordinates = model.predict(expdata)

ImSpec models
^^^^^^^^^^^^^^
AtomAI also provides models that can be used for predicting spectra from image data and vice versa. These models can be used for predicting property from structure. An example can be predicting approximate scanning tulleling spectroscopy or electron energy loss spectroscopy spectra from structural images of local sample regions (the assumption is of course that there is only a small variability of spectral behaviour within each  (sub)-image). The training/prediction routines are the same as for the semantic segmentation:

>>> in_dim = (16, 16)  # Input dimensions (image height and width)
>>> out_dim = (64,)  # Output dimensions (spectra length)
>>>
>>> # Initialize and train model
>>> model = aoi.models.ImSpec(in_dim, out_dim, latent_dim=10)
>>> model.fit(imgs_train, spectra_train, imgs_test, spectra_test,  # training data (numpy arrays)
>>>       full_epoch=True, training_cycles=120, swa=True)  # training parameters

Make a prediction with the trained ImSpec model by running

>>> prediction = model.predict(imgs_val, norm=False)

Deep ensembles
^^^^^^^^^^^^^^^

One can also use AtomAI to train an ensemble of models instead of just a single model. The average ensemble prediction is usually more accurate and reliable than that of the single model. In addition, we also get the information about the `uncertainty in our prediction <https://arxiv.org/abs/1612.01474>`_ for each pixel/point.

>>> # Ititialize and compile ensemble trainer
>>> etrainer = aoi.trainers.EnsembleTrainer("Unet", batch_norm=True, nb_classes=3, with_dilation=False)
>>> etrainer.compile_ensemble_trainer(training_cycles=500, compute_accuracy=True, swa=True)

>>> # Train ensemble of models starting every time with new randomly initialized weights
>>> smodel, ensemble = etrainer.train_ensemble_from_scratch(
>>>     images, labels, images_test, labels_test, n_models=10)

The ensemble of models can be then used to make a prediction with uncertainty estimates for each point (e.g. each pixel in the image):

>>> predictor = aoi.predictors.EnsemblePredictor(smodel, ensemble, nb_classes=3)
>>> nn_out_mean, nn_out_var = predictor.predict(expdata)

Variational autoencoders (VAE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

AtomAI also has built-in `variational autoencoders (VAEs) <https://arxiv.org/abs/1906.02691>`_ for finding in the unsupervised fashion the most effective reduced representation of system's local descriptors. The available VAEs are regular VAE, rotationally and/or translationally invariant VAE (rVAE), and class-conditined VAE/rVAE. The VAEs can be applied to both raw data and NN output, but typically work better with the latter. Here's a simple example:

>>> # Get a stack of subimages from experimental data (e.g. a semantically segmented atomic movie)
>>> imstack, com, frames = utils.extract_subimages(nn_output, coords, window_size=32)

>>> # Intitialize rVAE model
>>> input_dim = (32, 32)
>>> rvae = aoi.models.rVAE(input_dim) 

>>> # Train
>>> rvae.fit(
>>>    imstack_train, latent_dim=2,
>>>    rotation_prior=np.pi/3, training_cycles=100,
>>>    batch_size=100)   
>>> # Visualize the learned manifold
>>> rvae.manifold2d()

One can also use the trained VAE to view the data distribution in the latent space. In this example the first 3 latent variables are associated with rotations and xy-translations (they are automatically added in rVAE to whatever number of latent dimensions is specified), whereas the last 2 latent variables are associated with images content.

>>> encoded_mean, encoded_sd = rvae.encode(imstack)
>>> z1, z2, z3 = encoded_mean[:,0], encoded_mean[:, 1:3], encoded_mean[:, 3:]

Custom models
^^^^^^^^^^^^^^

Finally, it is possible to use AtomAI trainers and predictors for easy work with custom PyTorch models. Suppose we define a custom Pytorch neural network as

>>> # Here ConvBlock and UpsampleBlock are from atomai.nets module
>>> torch_encoder = torch.nn.Sequential(
>>>    ConvBlock(ndim=2, nb_layers=1, input_channels=1, output_channels=8, batch_norm=True),
>>>    torch.nn.MaxPool2d(2, 2),
>>>    ConvBlock(2, 2, 8, 16, batch_norm=False),
>>>    torch.nn.MaxPool2d(2, 2),
>>>    ConvBlock(2, 2, 16, 32, batch_norm=False),
>>>    torch.nn.MaxPool2d(2, 2),
>>>    ConvBlock(2, 2, 32, 64, batch_norm=False))
>>>
>>> torch_decoder = torch.nn.Sequential(
>>>    UpsampleBlock(ndim=2, input_channels=64, output_channels=64, mode="nearest"),
>>>    ConvBlock(2, 2, 64, 32, batch_norm=False),
>>>    UpsampleBlock(2, 32, 32, mode="nearest"),
>>>    ConvBlock(2, 2, 32, 16, batch_norm=False),
>>>    UpsampleBlock(2, 16, 16, mode="nearest"),
>>>    ConvBlock(2, 1, 16, 8, batch_norm=False),
>>>    torch.nn.Conv2d(8, 1, 1))
>>>
>>> torch_DAE = torch.nn.Sequential(torch_encoder, torch_decoder)

We can easily train this model using AtomAI's trainers:

>>> # Initialize trainer and pass our model to it
>>> trainer = aoi.trainers.BaseTrainer()
>>> trainer.set_model(torch_DAE)
>>> # Fix the initialization parameters (for reproducibility)
>>> set_train_rng(1)
>>> trainer._reset_weights() # start each time with the same initialization
>>> trainer._reset_training_history()
>>> # Compile trainer
>>> trainer.compile_trainer(
>>>    (imgdata_noisy, imgdata, imgdata_noisy_test, imgdata_test), # training data
>>>    loss="mse", training_cycles=500, swa=True)  # training parameters
>>> # Train
>>> trained_model = trainer.run()

The trained model can be used to make predictions on new data using AtomAI's predictors:

>>> p = aoi.predictors.BasePredictor(trained_model, use_gpu=True)
>>> prediction = p.predict(imgdata_noisy_test)

Not just deep learning
^^^^^^^^^^^^^^^^^^^^^^^

The information extracted by *atomnet* can be used for statistical analysis of raw and "decoded" data. For example, for a single atom-resolved image of ferroelectric material, one can identify domains with different ferroic distortions:

>>> # Get local descriptors
>>> imstack = aoi.stat.imlocal(nn_output, coordinates, window_size=32, coord_class=1)
>>> # Compute distortion "eigenvectors" with associated loading maps and plot results:
>>> pca_results = imstack.imblock_pca(n_components=4, plot_results=True)

For movies, one can extract trajectories of individual defects and calculate the transition probabilities between different classes:

>>> # Get local descriptors (such as subimages centered around impurities)
>>> imstack = aoi.stat.imlocal(nn_output, coordinates, window_size=32, coord_class=1)
>>> # Calculate Gaussian mixture model (GMM) components
>>> components, imgs, coords = imstack.gmm(n_components=10, plot_results=True)
>>> # Calculate GMM components and transition probabilities for different trajectories
>>> transitions_dict = imstack.transition_matrix(n_components=10, rmax=10)
>>> # and more

Installation
------------

First, install `PyTorch <https://pytorch.org/get-started/locally/>`_. Then, install AtomAI with

>>> pip install atomai
