"""Dataset classes.  *(brief §2)*

``SyntheticEnhanceDataset``
    Composites a fresh sample per ``__getitem__``: 256x256 patches cut from pages
    rectified at 1024x1448 for training, whole pages for evaluation. A practically
    infinite training set that never touches disk.
``SyntheticCornerDataset``
    The raw synthetic photo plus its corner labels, both as normalised coordinates
    and as four Gaussian heatmaps, so approaches A and B train off one dataset.
``FrozenSyntheticDataset``
    Reads the validation and test samples that were generated once with a fixed
    seed. Freezing them is what makes the validation curve measure the model
    instead of the dice.
``RealPhotoDataset``
    The evaluation-only bucket. ``mode="corner"`` yields the raw photo with its
    annotated corners scaled by the same factors; ``mode="enhance"`` yields the
    photo rectified with those annotated corners, alongside the reference scan at
    matching size. The degradation pipeline never runs on these — they arrive
    degraded by reality.

Corners are normalised to [0, 1] by image width and height, which makes the
detection task resolution-independent, and are *always* rescaled together with
their image: a corner label that is not transformed with its image is a wrong label.
"""
