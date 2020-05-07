Utility Functions
=============================


Training data preparation
-------------------------
.. autoclass:: atomai.utils.augmentor
    :members:
    :undoc-members:
    :member-order: bysource
.. autofunction:: atomai.utils.create_lattice_mask
.. autofunction:: atomai.utils.extract_patches_
.. autoclass:: atomai.utils.MakeAtom
    :members:
    :undoc-members:
    :member-order: bysource
.. autofunction:: atomai.utils.FFTmask
.. autofunction:: atomai.utils.FFTsub
.. autofunction:: atomai.utils.threshImg


Image preprocessing
-------------------

.. autofunction:: atomai.utils.preprocess_training_data
.. autofunction:: atomai.utils.torch_format
.. autofunction:: atomai.utils.img_resize
.. autofunction:: atomai.utils.img_pad


Atomic Coordinates
-------------------

.. autofunction:: atomai.utils.find_com
.. autofunction:: atomai.utils.get_nn_distances
.. autofunction:: atomai.utils.get_nn_distances_
.. autofunction:: atomai.utils.peak_refinement
.. autofunction:: atomai.utils.compare_coordinates
.. autofunction:: atomai.utils.filter_cells
.. autofunction:: atomai.utils.filter_cells_
.. autofunction:: atomai.utils.cv_thresh


Visualization
--------------
.. autofunction:: atomai.utils.plot_trajectories
.. autofunction:: atomai.utils.plot_transitions
.. autofunction:: atomai.utils.plot_trajectories_transitions


Trained weights
----------------
.. autofunction:: atomai.utils.load_weights
.. autofunction:: atomai.utils.average_weights
.. autofunction:: atomai.utils.nb_filters_classes
