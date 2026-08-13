"""Network architectures.  *(brief §3.1 and §5)*

The brief names this file explicitly, so every architecture lives here.

Designed from scratch — no imported U-Net, no pre-trained weights. The first
version of every model carries no dropout and no other explicit regularisation
either; that arrives later, and only as dropout, so the regularisation study has
exactly one variable in it.

Built so far:

``ConvBlock``
    (Conv3x3 -> Norm -> ReLU) x2, the shared building block.
``DocUNet``
    The enhancement network. Encoder 32/64/128/256, 512 bottleneck, maxpool down,
    transposed-conv up, **concatenated skip connections** — text strokes are thin
    and do not survive a bottleneck without them. Fully convolutional, so training
    can happen on 256x256 patches while inference runs on a whole page.
``CornerRegNet``
    Corner detection, approach A *(brief §5)*. A five-stage convolutional encoder,
    **flattened** into two fully connected layers that emit the eight numbers of
    the four corners. Flattened rather than globally pooled: global average
    pooling discards where in the frame each activation fired, which is exactly
    the information a coordinate is made of.
``CornerHeatNet``
    Corner detection, approach B. The same encoder-decoder family as the
    enhancement network, stopping one level short of full resolution to emit four
    Gaussian heatmaps at half the input size.
``soft_argmax2d`` / ``heatmap_peaks``
    Reading coordinates back off those heatmaps: a differentiable sub-pixel
    expectation, and the plain arg-max it is compared against.
``build_model`` / ``load_model``
    A model from a config block, and a model from a checkpoint written by the
    trainer. Every checkpoint stores the config it was built from, so nothing has
    to be remembered to reload one.

Every model declares an ``output_kind`` — ``"restoration"``, ``"coords"`` or
``"heatmaps"``. The trainer, the losses and the metrics all dispatch on it, so
adding a model is one class and one registry entry rather than an edit in four
files.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    "ConvBlock",
    "DocUNet",
    "CornerRegNet",
    "CornerHeatNet",
    "soft_argmax2d",
    "heatmap_peaks",
    "corners_from_output",
    "build_model",
    "load_model",
    "count_parameters",
    "clamp_image",
]


def clamp_image(tensor: torch.Tensor) -> torch.Tensor:
    """Bring a prediction back into [0, 1] before it is measured or saved.

    A sigmoid head is already inside the range and this is a no-op for it. A
    residual head is not — it predicts a correction to add to the input, and
    nothing stops the sum from leaving the range — so the clamp lives at the
    boundary where the image becomes a number or a file, and never inside the
    loss, where a hard clamp would silently zero the gradient of every pixel that
    overshot.
    """
    return tensor.clamp(0.0, 1.0)


def pad_to_multiple(x: torch.Tensor, divisor: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad up to a multiple of the downsampling factor, by replication.

    Returns the padded tensor and the size that came in, so the caller can crop
    its output back. 1024x1448 is not a multiple of 16, and neither is an
    arbitrary page a grader hands the model on the day. Padding by replication
    rather than with zeros keeps a black frame from appearing at the edge of the
    receptive field and being mistaken for content.
    """
    height, width = x.shape[-2:]
    pad_h = (-height) % divisor
    pad_w = (-width) % divisor
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x, (height, width)


def _norm(kind: str, channels: int) -> nn.Module:
    """Normalisation by name, so the config can turn it off for the ablation."""
    if kind in (None, "none"):
        return nn.Identity()
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if kind == "group":
        # Eight groups divides every channel count this network uses.
        return nn.GroupNorm(min(8, channels), channels)
    raise ValueError(f"unknown norm {kind!r}; expected batch, instance, group or none")


class ConvBlock(nn.Module):
    """(Conv3x3 -> Norm -> ReLU) x2 — the unit both networks are built from.

    The convolutions carry no bias when they are followed by a normalisation
    layer, which would subtract it again on the next line.

    ``dropout`` is wired through but zero for the first version of every model,
    per the brief's §3.1 constraint. When it is switched on for the regularisation
    study it applies ``Dropout2d``, which drops whole feature *channels* rather
    than scattered pixels: neighbouring activations in a convolutional map are
    strongly correlated, so dropping individual ones leaks most of the signal
    through anyway and regularises far less than the same rate suggests.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str = "batch",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        bias = norm in (None, "none")
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=bias),
            _norm(norm, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=bias),
            _norm(norm, out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DocUNet(nn.Module):
    """The enhancement network: degraded rectified page in, clean scan out.

    An encoder-decoder in the shape the brief describes *(brief §3.1)*. The
    encoder halves the resolution four times, which is what lets a 3x3 kernel
    near the bottleneck see a quarter of the patch at once — the receptive field
    a shadow or an illumination gradient needs before it can be recognised as one.
    The decoder puts the resolution back with transposed convolutions.

    **The skip connections are the point.** A pen stroke is two or three pixels
    wide; sixteen-fold downsampling leaves nothing of it to reconstruct from. Each
    decoder stage is therefore handed the encoder feature map at its own
    resolution, concatenated channel-wise, so the fine detail never has to survive
    the bottleneck — only the *context* does. ``skips=False`` exists to
    demonstrate that in the report rather than assert it.

    Two heads, chosen by ``residual_output``:

    * direct — a 1x1 convolution and a sigmoid, so the output is an image in
      [0, 1] by construction;
    * residual — the network predicts the *correction* and the input is added
      back. Most of a document photo is already correct, so this starts the
      network at "change nothing" instead of at "invent a page", and thin strokes
      tend to survive better because the network is not re-deriving them from
      scratch. The correction goes through a tanh: the largest change any pixel
      can legitimately need is ±1, which is exactly that function's range.

    Fully convolutional, and it pads internally to a multiple of the downsampling
    factor, so it accepts any input size — 256x256 patches while training,
    1024x1448 pages or 512x512 tiles at inference, with the output cropped back to
    the size that came in.
    """

    output_kind = "restoration"

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base: int = 32,
        depth: int = 4,
        norm: str = "batch",
        dropout: float = 0.0,
        bottleneck_dropout: float | None = None,
        residual_output: bool = False,
        skips: bool = True,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")
        self.depth = int(depth)
        self.base = int(base)
        self.residual_output = bool(residual_output)
        self.skips = bool(skips)
        self.divisor = 2**self.depth

        widths = [base * 2**level for level in range(depth)]  # 32, 64, 128, 256
        bottleneck_width = base * 2**depth  # 512

        self.encoder = nn.ModuleList()
        channels = in_channels
        for width in widths:
            self.encoder.append(ConvBlock(channels, width, norm=norm, dropout=dropout))
            channels = width
        self.pool = nn.MaxPool2d(2)

        drop_bottleneck = dropout if bottleneck_dropout is None else bottleneck_dropout
        self.bottleneck = ConvBlock(channels, bottleneck_width, norm=norm, dropout=drop_bottleneck)

        self.upsample = nn.ModuleList()
        self.decoder = nn.ModuleList()
        channels = bottleneck_width
        for width in reversed(widths):
            self.upsample.append(nn.ConvTranspose2d(channels, width, kernel_size=2, stride=2))
            merged = width * 2 if self.skips else width
            self.decoder.append(ConvBlock(merged, width, norm=norm, dropout=dropout))
            channels = width

        self.head = nn.Conv2d(channels, out_channels, kernel_size=1)
        self._initialise()

    def _initialise(self) -> None:
        """He initialisation, which is the one that matches ReLU.

        The head is started small on purpose: a residual model then begins as
        approximately the identity, and a direct one begins at a flat mid-grey
        rather than at saturated noise that the sigmoid would take a while to
        climb out of.
        """
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        source = x
        x, (height, width) = pad_to_multiple(x, self.divisor)

        features = []
        for block in self.encoder:
            x = block(x)
            features.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up, block, skip in zip(self.upsample, self.decoder, reversed(features)):
            x = up(x)
            if self.skips:
                x = torch.cat([x, skip], dim=1)
            x = block(x)

        x = self.head(x)
        x = x[..., :height, :width]

        if self.residual_output:
            return source + torch.tanh(x)
        return torch.sigmoid(x)


# ---------------------------------------------------------------------------
# corner detection, approach A: direct coordinate regression  (brief §5)
# ---------------------------------------------------------------------------
#: Where an average page's corners sit in a photo, as normalised (x, y). The
#: regression head's bias starts here, so the untrained network answers "a page,
#: roughly centred, filling about half the frame" instead of answering noise.
PRIOR_QUAD = ((0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75))


class CornerRegNet(nn.Module):
    """A photo in, eight numbers out: the four page corners *(brief §5, approach A)*.

    Five convolutional stages halve the resolution each time — 256 to 8 — and the
    result is **flattened** into two fully connected layers. Flattened, not
    globally pooled, and that is the one decision in this class worth defending:
    global average pooling answers "is there a corner-like thing in this image",
    which is a classification question. A coordinate is made of *where* the
    activation fired, and pooling is precisely the operation that throws that
    away. The cost is that the network is no longer size-agnostic and that most
    of its parameters end up in one matrix.

    That matrix is also the interesting difficulty. It has to learn the entire map
    from feature position to coordinate from scratch: nothing in the architecture
    knows that a page ten pixels further right should produce an answer ten pixels
    larger. The heatmap formulation gets that for free, which is the heart of the
    comparison the brief asks for.

    Two heads, chosen by ``output_activation``:

    * ``sigmoid`` — coordinates are confined to [0, 1] by construction, which is
      where every label lives. The bias is initialised so the network starts at
      :data:`PRIOR_QUAD` rather than at all four corners piled on the centre.
    * ``none`` — the raw output, for the ablation that asks whether the sigmoid's
      saturation is holding training back.

    ``fc_dropout`` is the classic place for dropout and the brief names it as
    such *(brief §6)*. It is 0.0 here and stays that way until the study that is
    about it.
    """

    output_kind = "coords"

    def __init__(
        self,
        in_channels: int = 3,
        corners: int = 4,
        base: int = 32,
        stages: int = 5,
        grid: int = 8,
        hidden: int = 512,
        norm: str = "batch",
        dropout: float = 0.0,
        fc_dropout: float = 0.0,
        output_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        if stages < 1:
            raise ValueError(f"stages must be at least 1, got {stages}")
        if output_activation not in ("sigmoid", "none"):
            raise ValueError(f"unknown output_activation {output_activation!r}")
        self.corners = int(corners)
        self.grid = int(grid)
        self.output_activation = output_activation

        # 32, 64, 128, 256, 256 — the width stops doubling at the last stage
        # because the fully connected layer that follows is already the largest
        # thing in the model and doubling the map it reads doubles it again.
        widths = [base * 2 ** min(level, 3) for level in range(stages)]
        self.encoder = nn.ModuleList()
        channels = in_channels
        for width in widths:
            self.encoder.append(ConvBlock(channels, width, norm=norm, dropout=dropout))
            channels = width
        self.pool = nn.MaxPool2d(2)

        # An identity at the size this trains at (256 -> 8 after five halvings),
        # and a guard that keeps the flatten well defined if something hands the
        # network a different size at inference.
        self.collapse = nn.AdaptiveAvgPool2d((self.grid, self.grid))

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * self.grid * self.grid, hidden),
            nn.ReLU(inplace=True),
            *([nn.Dropout(fc_dropout)] if fc_dropout > 0 else []),
            nn.Linear(hidden, self.corners * 2),
        )
        self._initialise()

    def _initialise(self) -> None:
        """He initialisation, and a final bias that starts at an average page."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

        final = self.head[-1]
        nn.init.normal_(final.weight, std=1e-3)
        prior = torch.tensor(PRIOR_QUAD, dtype=torch.float32).flatten()
        if self.output_activation == "sigmoid":
            prior = torch.log(prior / (1.0 - prior))  # the logit of where it should start
        with torch.no_grad():
            final.bias.copy_(prior[: final.bias.numel()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.encoder:
            x = self.pool(block(x))
        x = self.collapse(x)
        x = self.head(x)
        if self.output_activation == "sigmoid":
            x = torch.sigmoid(x)
        return x.view(x.shape[0], self.corners, 2)


# ---------------------------------------------------------------------------
# corner detection, approach B: heatmap regression  (brief §5)
# ---------------------------------------------------------------------------
class CornerHeatNet(nn.Module):
    """A photo in, four Gaussian heatmaps out *(brief §5, approach B)*.

    The same encoder-decoder machinery as the enhancement network, which is what
    the brief suggests reusing, with two changes. The head emits one channel per
    corner instead of three colour channels, and the decoder stops one level
    short of the input resolution: 256 in, 128 out. Predicting at half resolution
    costs a quarter of the decoder's work and gives up nothing that matters,
    because the coordinate is read back with a sub-pixel expectation rather than
    with the index of the brightest cell.

    **Why this formulation is expected to behave differently.** Every output cell
    is spatially aligned with the input, so "there is a corner here" stays a local
    decision made on local evidence, and translation equivariance comes free from
    the convolutions. The supervision is also denser by four orders of magnitude —
    4 x 128 x 128 supervised values per sample against eight. What it gives up is
    a continuous output: the map is a grid, and everything sub-pixel has to come
    out of :func:`soft_argmax2d`.

    The head is **linear by default**, not squashed through a sigmoid. The
    targets are Gaussians in [0, 1], so a sigmoid is tempting, but MSE against a
    saturating output learns slowly in exactly the regions that matter — the peak
    and the far background — and heatmap regression in the literature is
    overwhelmingly done with a linear head. ``head_activation: sigmoid`` is there
    for the ablation.
    """

    output_kind = "heatmaps"

    def __init__(
        self,
        in_channels: int = 3,
        corners: int = 4,
        base: int = 32,
        depth: int = 4,
        out_stride: int = 2,
        norm: str = "batch",
        dropout: float = 0.0,
        bottleneck_dropout: float | None = None,
        head_activation: str = "none",
        skips: bool = True,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")
        if out_stride < 1 or (out_stride & (out_stride - 1)):
            raise ValueError(f"out_stride must be a power of two, got {out_stride}")
        levels_kept = int(out_stride).bit_length() - 1
        if levels_kept > depth:
            raise ValueError(f"out_stride {out_stride} needs more than {depth} encoder levels")
        if head_activation not in ("none", "sigmoid"):
            raise ValueError(f"unknown head_activation {head_activation!r}")

        self.depth = int(depth)
        self.base = int(base)
        self.corners = int(corners)
        self.out_stride = int(out_stride)
        self.head_activation = head_activation
        self.skips = bool(skips)
        self.divisor = 2**self.depth

        widths = [base * 2**level for level in range(depth)]  # 32, 64, 128, 256
        bottleneck_width = base * 2**depth  # 512

        self.encoder = nn.ModuleList()
        channels = in_channels
        for width in widths:
            self.encoder.append(ConvBlock(channels, width, norm=norm, dropout=dropout))
            channels = width
        self.pool = nn.MaxPool2d(2)

        drop_bottleneck = dropout if bottleneck_dropout is None else bottleneck_dropout
        self.bottleneck = ConvBlock(channels, bottleneck_width, norm=norm, dropout=drop_bottleneck)

        # One upsampling step per encoder level the output is *not* stopping at.
        self.skip_levels = [depth - 1 - step for step in range(depth - levels_kept)]
        self.upsample = nn.ModuleList()
        self.decoder = nn.ModuleList()
        channels = bottleneck_width
        for level in self.skip_levels:
            width = widths[level]
            self.upsample.append(nn.ConvTranspose2d(channels, width, kernel_size=2, stride=2))
            merged = width * 2 if self.skips else width
            self.decoder.append(ConvBlock(merged, width, norm=norm, dropout=dropout))
            channels = width

        self.head = nn.Conv2d(channels, self.corners, kernel_size=1)
        self._initialise()

    def _initialise(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        # A near-flat initial map. Started large, the softmax-like extraction picks
        # a confident answer out of noise and the first gradients fight it.
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, (height, width) = pad_to_multiple(x, self.divisor)

        features = []
        for block in self.encoder:
            x = block(x)
            features.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up, block, level in zip(self.upsample, self.decoder, self.skip_levels):
            x = up(x)
            if self.skips:
                x = torch.cat([x, features[level]], dim=1)
            x = block(x)

        x = self.head(x)
        # Crop the padding back off, in output cells rather than input pixels.
        x = x[..., : -(-height // self.out_stride), : -(-width // self.out_stride)]
        return torch.sigmoid(x) if self.head_activation == "sigmoid" else x


# ---------------------------------------------------------------------------
# reading coordinates off a heatmap
# ---------------------------------------------------------------------------
def _coordinate_grids(height: int, width: int, device, dtype):
    """Cell centres in normalised coordinates, matching ``normalize_corners``.

    ``(i + 0.5) / n``, not ``i / n``: that is the convention the labels are stored
    in and the only one that survives a resize, so a coordinate read off a heatmap
    is directly comparable with the label it is scored against.
    """
    xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    return xs, ys


def heatmap_peaks(heatmaps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Plain arg-max: ``(coords (B, K, 2) normalised, peak values (B, K))``.

    Not differentiable, and quantised to the cell grid — at 128 cells for a 256
    input that is two pixels, which is the same order as the error being measured.
    It is here because it is the baseline :func:`soft_argmax2d` has to beat, and
    because the peak value is a usable confidence for the inference pipeline.
    """
    if heatmaps.dim() != 4:
        raise ValueError(f"expected (B, K, H, W) heatmaps, got {tuple(heatmaps.shape)}")
    batch, count, height, width = heatmaps.shape
    flat = heatmaps.flatten(2)
    values, indices = flat.max(dim=-1)
    xs, ys = _coordinate_grids(height, width, heatmaps.device, heatmaps.dtype)
    rows = torch.div(indices, width, rounding_mode="floor")
    coords = torch.stack([xs[indices % width], ys[rows]], dim=-1)
    return coords.view(batch, count, 2), values


#: The neighbourhood :func:`soft_argmax2d` reads a coordinate out of, in heatmap
#: cells, everywhere a *reported* number is produced. Not a tuning knob — it is
#: the difference between measuring the network and measuring its background
#: noise. See :func:`corners_from_output`.
SOFT_ARGMAX_WINDOW = 11


def soft_argmax2d(
    heatmaps: torch.Tensor,
    mode: str = "normalised",
    beta: float = 1.0,
    window: int | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Sub-pixel coordinates from heatmaps, differentiably. ``(B, K, H, W)`` -> ``(B, K, 2)``.

    The centre of mass of each map, in the same normalised coordinates the labels
    use. Differentiable, because the coordinate is a weighted sum of the map's own
    values — which is what lets a coordinate loss be applied to a heatmap model,
    and what the bonus end-to-end fine-tuning would need.

    Two ways to turn a map into weights:

    * ``"normalised"`` (the default) treats the map as an unnormalised density:
      clamp the negatives away and divide by the total. Applied to a *target*
      heatmap this returns the label it was drawn from almost exactly, which is
      the property that makes it testable — an extraction that cannot invert its
      own targets has no business being trusted on predictions.
    * ``"softmax"`` weights by ``softmax(beta * h)``. Scale-free and always
      positive, but the answer depends on ``beta``, and at any finite ``beta`` a
      flat map pulls the estimate toward the centre of the frame.

    ``window`` restricts the expectation to a square neighbourhood of the peak.
    That is what makes the estimate robust to a *second* mode — the generator puts
    a distractor sheet in a fifth of its corner samples, and a global expectation
    over two blobs returns a point on neither page. Off during training, where the
    gradient should reach the whole map; on at inference, where only the answer
    matters.
    """
    if heatmaps.dim() != 4:
        raise ValueError(f"expected (B, K, H, W) heatmaps, got {tuple(heatmaps.shape)}")
    batch, count, height, width = heatmaps.shape
    maps = heatmaps.float()

    if mode == "softmax":
        weights = torch.softmax(beta * maps.flatten(2), dim=-1).view(batch, count, height, width)
    elif mode == "normalised":
        weights = maps.clamp_min(0.0)
    else:
        raise ValueError(f"mode must be 'normalised' or 'softmax', not {mode!r}")

    if window:
        radius = int(window) // 2
        peaks = maps.flatten(2).argmax(dim=-1)
        peak_y = torch.div(peaks, width, rounding_mode="floor")[..., None, None]
        peak_x = (peaks % width)[..., None, None]
        rows = torch.arange(height, device=maps.device)[None, None, :, None]
        columns = torch.arange(width, device=maps.device)[None, None, None, :]
        near = ((rows - peak_y).abs() <= radius) & ((columns - peak_x).abs() <= radius)
        weights = weights * near

    total = weights.flatten(2).sum(dim=-1)
    # A map with no positive mass says nothing about where its corner is. The
    # honest answer is the middle of the frame, and saying so beats dividing by a
    # clamped zero and returning the top-left cell as if it meant something.
    weights = weights / total[..., None, None].clamp_min(eps)
    xs, ys = _coordinate_grids(height, width, maps.device, maps.dtype)
    x = (weights.sum(dim=-2) * xs).sum(dim=-1)
    y = (weights.sum(dim=-1) * ys).sum(dim=-1)
    coords = torch.stack([x, y], dim=-1)
    return torch.where(total[..., None] > eps, coords, torch.full_like(coords, 0.5))


def corners_from_output(
    output: torch.Tensor,
    window: int | None = SOFT_ARGMAX_WINDOW,
) -> torch.Tensor:
    """Normalised ``(B, 4, 2)`` corners from whatever a detector emitted.

    One function, called by the trainer's validation, by the evaluation table and
    by the inference pipeline, because they were three call sites that could
    disagree — and did.

    **The window is not a tuning knob.** Measured on the first trained heatmap
    detector, over the 200-photo synthetic test bucket, the same weights scored:

    ==========================  =========  ========
    extraction                  mean (px)  PCK @ 1%
    ==========================  =========  ========
    global centre of mass            6.83     0.000
    plain arg-max                    1.38     0.915
    windowed centre of mass          1.06     0.890
    ==========================  =========  ========

    A factor of six, from the same network. The head is linear, so its background
    is not zero but small positive *noise* — and spread over 16 384 cells, a
    little noise everywhere carries more mass than the blob does. A global centre
    of mass therefore measures mostly the background and reports something near
    the middle of the frame, which is why the error had a floor no amount of
    training could push through: no sample ever landed all four corners inside 1%
    of the diagonal, at any epoch.

    Passing ``window=None`` restores the global expectation. That is the right
    thing while *training* a coordinate term through this function — the gradient
    should reach the whole map, including the background it needs to learn to
    suppress — and the wrong thing everywhere a number is reported.
    """
    if output.dim() == 3:
        return output
    if output.dim() == 4:
        return soft_argmax2d(output, window=window)
    raise ValueError(f"expected (B, K, 2) or (B, K, H, W), got {tuple(output.shape)}")


# ---------------------------------------------------------------------------
# building and loading
# ---------------------------------------------------------------------------
MODELS = {
    "docunet": DocUNet,
    "cornerregnet": CornerRegNet,
    "cornerheatnet": CornerHeatNet,
}


def build_model(config) -> nn.Module:
    """Instantiate the model described by a config's ``model:`` block.

    The block is passed through to the constructor as keyword arguments, so an
    unknown key fails here, loudly, rather than being silently ignored and
    leaving a run that did not test what its filename says it tested.
    """
    settings = dict(config.get("model") or {})
    name = str(settings.pop("name", "docunet")).lower()
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; expected one of {sorted(MODELS)}")
    try:
        return MODELS[name](**settings)
    except TypeError as exc:
        raise TypeError(f"bad model: settings for {name!r}: {exc}") from exc


def load_model(checkpoint, device=None, weights: str = "model"):
    """Rebuild a trained model from a checkpoint written by the trainer.

    Returns ``(model, config)``. The checkpoint carries its own config, so
    reloading a model never depends on remembering which config file produced it
    — which is the failure mode that makes half-year-old checkpoints useless.
    """
    from .config import Config

    if isinstance(checkpoint, (str,)) or hasattr(checkpoint, "__fspath__"):
        checkpoint = torch.load(str(checkpoint), map_location="cpu", weights_only=False)

    config = Config(checkpoint["config"])
    model = build_model(config)
    model.load_state_dict(checkpoint[weights])
    if device is not None:
        model.to(device)
    model.eval()
    return model, config


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Parameter count, for the report's architecture table."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
