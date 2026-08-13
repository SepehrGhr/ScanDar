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
``build_model`` / ``load_model``
    A model from a config block, and a model from a checkpoint written by the
    trainer. Every checkpoint stores the config it was built from, so nothing has
    to be remembered to reload one.

Not built yet: the two corner detectors. Approach A is a conv encoder followed by
fully connected layers emitting the eight numbers of the four corners — global
pooling would throw away exactly the spatial information coordinates are made of,
so the encoder output gets flattened instead. Approach B reuses this file's
encoder-decoder trunk to emit four Gaussian heatmaps, read back with a
differentiable sub-pixel soft-argmax. They are written when the corner-detection
task is built; the enhancement network came first because the corner labels are
not yet exported from the annotation tool.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    "ConvBlock",
    "DocUNet",
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

    # -- size handling ------------------------------------------------------
    def _pad(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """Pad up to a multiple of the downsampling factor, by replication.

        1024x1448 is not a multiple of 16, and neither is an arbitrary page a
        grader hands the model on the day. Padding by replication rather than
        with zeros keeps a black frame from appearing at the edge of the receptive
        field and being mistaken for content.
        """
        height, width = x.shape[-2:]
        pad_h = (-height) % self.divisor
        pad_w = (-width) % self.divisor
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x, (height, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        source = x
        x, (height, width) = self._pad(x)

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
# building and loading
# ---------------------------------------------------------------------------
MODELS = {"docunet": DocUNet}


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
