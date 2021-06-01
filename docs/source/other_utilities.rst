Other utilities
================

Statistics
----------
.. autoclass:: atomai.stat.imlocal
    :members:
    :undoc-members:
    :member-order: bysource
    
.. autofunction:: atomai.stat.update_classes

Image transforms
----------------
.. autoclass:: atomai.transforms.datatransform
    :members:
    :undoc-members:
    :member-order: bysource

Training data preparation
-------------------------

.. autofunction:: atomai.utils.create_lattice_mask
.. autofunction:: atomai.utils.create_multiclass_lattice_mask
.. autofunction:: atomai.utils.extract_patches
.. autofunction:: atomai.utils.extract_random_subimages
.. autofunction:: atomai.utils.extract_subimages

.. autoclass:: atomai.utils.MakeAtom
    :members:
    :undoc-members:
    :member-order: bysource

.. autofunction:: atomai.utils.FFTmask
.. autofunction:: atomai.utils.FFTsub
.. autofunction:: atomai.utils.threshImg


Image pre/post processing
-------------------------

.. autofunction:: atomai.utils.torch_format
.. autofunction:: atomai.utils.img_resize
.. autofunction:: atomai.utils.img_pad
.. autofunction:: atomai.utils.crop_borders
.. autofunction:: atomai.utils.filter_cells
.. autofunction:: atomai.utils.get_blob_params
.. autofunction:: atomai.utils.cv_thresh
.. autofunction:: atomai.utils.cv_resize
.. autofunction:: atomai.utils.cv_resize_stack


Atomic Coordinates
------------------

.. autofunction:: atomai.utils.map_bonds
.. autofunction:: atomai.utils.find_com
.. autofunction:: atomai.utils.get_nn_distances
.. autofunction:: atomai.utils.get_nn_distances_
.. autofunction:: atomai.utils.peak_refinement
.. autofunction:: atomai.utils.compare_coordinates


Visualization
--------------
.. autofunction:: atomai.utils.plot_trajectories
.. autofunction:: atomai.utils.plot_transitions
.. autofunction:: atomai.utils.plot_trajectories_transitions

ASE utilities
-------------
.. autofunction:: atomai.utils.ase_obj_basic
.. autofunction:: atomai.utils.ase_obj_adv

Datasets
--------
.. autofunction:: atomai.utils.datasets.stem_smbfo
.. autofunction:: atomai.utils.datasets.stem_graphene

