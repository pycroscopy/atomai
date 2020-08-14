Multivariate analysis
======================

Analysis of local subimages
----------------------------
.. autoclass:: atomai.atomstat.imlocal
    :members:
    :undoc-members:
    :member-order: bysource

Transitions
------------
.. autofunction:: atomai.atomstat.calculate_transition_matrix
.. autofunction:: atomai.atomstat.sum_transitions

Update predicted classes
-------------------------
.. autofunction:: atomai.atomstat.update_classes


Variational autoencoders (VAEs)
===============================

.. autoclass:: atomai.atomstat.EncoderDecoder
    :members:
    :undoc-members:
    :member-order: bysource

.. autoclass:: atomai.atomstat.rVAE
    :members:
    :undoc-members:
    :member-order: bysource

.. autoclass:: atomai.atomstat.VAE
    :members:
    :undoc-members:
    :member-order: bysource

.. autofunction:: atomai.atomstat.rvae
.. autofunction:: atomai.atomstat.vae
.. autofunction:: atomai.atomstat.load_vae_model
