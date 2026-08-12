"""Network architectures.  *(Phases 2, 4 and 5; brief §3.1 and §5)*

The brief names this file explicitly, so every architecture lives here.

Designed from scratch — no imported U-Net, no pre-trained weights. The first
versions carry no dropout and no other explicit regularisation either; that
arrives, and only as dropout, in Phase 5.

``ConvBlock``
    (Conv3x3 -> Norm -> ReLU) x2, the shared building block.
``DocUNet``
    The enhancement network. Encoder 32/64/128/256, 512 bottleneck, maxpool down,
    transposed-conv up, **concatenated skip connections** — text strokes are thin
    and do not survive a bottleneck without them. Fully convolutional, so training
    can happen on 256x256 patches while inference runs on a whole page.
``CornerRegNet``
    Approach A: a conv encoder followed by fully connected layers emitting eight
    numbers, the normalised (x, y) of the four corners. Global pooling would throw
    away exactly the spatial information coordinates are made of, so the encoder
    output is flattened instead.
``CornerHeatNet``
    Approach B: the same encoder-decoder trunk emitting four heatmaps, one Gaussian
    blob per corner, with ``soft_argmax2d`` for differentiable sub-pixel extraction.
``soft_argmax2d``
    Spatial expectation over a heatmap — sub-pixel, and differentiable, which is
    what the bonus end-to-end fine-tuning needs.
"""
