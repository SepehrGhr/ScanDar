"""ScanDar demo — a phone photo in, a clean scan out  *(brief §7)*.

    python app/demo.py

Pick a model or a chain of models *before* dropping in a photo — that ordering
is deliberate, because the six choices are not interchangeable: three take a
raw photo (a corner detector alone, or a detector chained into the enhancer),
one takes an already-flattened page (the enhancer alone), and picking the wrong
one for the image on hand produces a confusing result rather than an error, so
the choice has to come first.

Every checkpoint referenced here is loaded from ``outputs/runs/<name>/best.pt``
through :func:`scandar.model.load_model`, on demand and cached in memory for the
life of the process — the same lookup ``scandar scan`` uses, so the demo can
never drift from what the CLI does.
"""

import os
from pathlib import Path

# gradio pulls in httpx, which at import time builds a client from the
# process's proxy env vars. This machine's `all_proxy` is `socks://...`, a
# scheme httpx does not recognise (it wants `socks5://`), so a plain `import
# gradio` raises before any of this module's own code runs — and even past
# that, `launch()`'s own startup self-check routes through the configured
# HTTP(S) proxy and 127.0.0.1 never answers there, so it also fails against a
# server that came up fine. The demo talks to nothing but localhost, so the
# fix is to make it not look.
for _proxy_var in (
    "all_proxy", "ALL_PROXY", "http_proxy", "HTTP_PROXY",
    "https_proxy", "HTTPS_PROXY", "ftp_proxy", "FTP_PROXY",
):
    os.environ.pop(_proxy_var, None)

import gradio as gr
import numpy as np

from scandar.device import get_device
from scandar.io import imwrite_rgb, paths
from scandar.model import load_model
from scandar.pipelines import detect_corners, draw_corners, enhance_document, scan_document

DEVICE = get_device()

#: Every choice the dropdown offers, and how to run it. "checkpoints" names
#: run directories under outputs/runs/ — resolved lazily, on first use, so the
#: app starts even if some of the six are missing on this machine.
TASKS = {
    "Corner detector — heatmap  (corner_heat)": {
        "kind": "detect", "detector": "corner_heat",
    },
    "Corner detector — regression  (corner_reg)": {
        "kind": "detect", "detector": "corner_reg",
    },
    "Enhancement only  (enhance_realistic)": {
        "kind": "enhance", "enhancer": "enhance_realistic",
    },
    "Full chain — heatmap detector + enhancer": {
        "kind": "chain", "detector": "corner_heat", "enhancer": "enhance_realistic",
    },
    "Full chain — regression detector + enhancer": {
        "kind": "chain", "detector": "corner_reg", "enhancer": "enhance_realistic",
    },
    "End-to-end fine-tuned scanner  (corner_heat_e2e)": {
        "kind": "e2e", "scanner": "corner_heat_e2e",
    },
}
DEFAULT_TASK = "Full chain — heatmap detector + enhancer"

TASK_HELP = {
    "detect": "Takes a **raw photo**. Draws the four detected corners; nothing is rectified.",
    "enhance": "Takes an **already-flattened page** (e.g. from a chain run's rectified output), "
               "not a raw photo — this model was never shown a photographed background.",
    "chain": "Takes a **raw photo**. Detects the corners, flattens the page, restores it — the "
             "two networks bolted together, run one after the other.",
    "e2e": "Takes a **raw photo**. Same chain as above, but the detector was fine-tuned through "
           "the warp by the enhancer's own loss — see docs/end-to-end-scanner.md for what that "
           "bought (a demonstrated mechanism, not a measurable improvement).",
}

_MODEL_CACHE: dict[str, tuple] = {}


def _load(name: str):
    if name not in _MODEL_CACHE:
        checkpoint = paths.runs / name / "best.pt"
        if not checkpoint.is_file():
            raise gr.Error(f"no checkpoint at {checkpoint} — train {name!r} first")
        _MODEL_CACHE[name] = load_model(checkpoint, device=DEVICE)
    return _MODEL_CACHE[name]


def _input_size(config) -> int:
    return int(config.get("data", {}).get("corner_input", 256))


def _source_note(result: dict) -> str:
    note = f"corners from the **{result['source']}** path"
    if result.get("problem"):
        note += f" — the model's own quad was rejected ({result['problem']})"
    if result.get("confidence") is not None:
        note += f", confidence {result['confidence']:.3f}"
    return note


def describe(task_label: str) -> str:
    return TASK_HELP[TASKS[task_label]["kind"]]


def run(task_label: str, photo):
    if photo is None:
        raise gr.Error("drop in a photo first")
    photo = np.asarray(photo)
    if photo.ndim != 3:
        raise gr.Error(f"expected a colour image, got shape {photo.shape}")
    if photo.shape[-1] == 4:
        photo = photo[..., :3]
    photo = np.ascontiguousarray(photo)

    spec = TASKS[task_label]
    kind = spec["kind"]
    corners_img = rectified_img = scan_img = None

    if kind == "detect":
        model, config = _load(spec["detector"])
        result = detect_corners(photo, model, device=DEVICE, input_size=_input_size(config))
        corners_img = draw_corners(photo, result["corners"])
        status = _source_note(result)
        final = corners_img

    elif kind == "enhance":
        model, _ = _load(spec["enhancer"])
        scan_img = enhance_document(photo, model, device=DEVICE)
        status = "enhanced directly — no detection or rectification in this path"
        final = scan_img

    else:  # "chain" or "e2e"
        if kind == "chain":
            detector, detector_config = _load(spec["detector"])
            enhancer, _ = _load(spec["enhancer"])
            input_size = _input_size(detector_config)
        else:
            scanner, scanner_config = _load(spec["scanner"])
            detector, enhancer = scanner, None
            input_size = _input_size(scanner_config)

        result = scan_document(photo, detector, enhancer, device=DEVICE, input_size=input_size)
        corners_img = draw_corners(photo, result["corners"])
        rectified_img = result["rectified"]
        scan_img = result["scan"]
        status = _source_note(result)
        final = scan_img

    output_path = Path(paths.out) / "demo" / "last_result.png"
    imwrite_rgb(output_path, final)

    return (
        gr.update(value=corners_img, visible=corners_img is not None),
        gr.update(value=rectified_img, visible=rectified_img is not None),
        gr.update(value=scan_img, visible=scan_img is not None),
        status,
        str(output_path),
    )


EXAMPLE_PHOTOS = sorted((paths.real_photos).glob("*.jpg"))[:6] if paths.real_photos.is_dir() else []

with gr.Blocks(title="ScanDar") as demo:
    gr.Markdown(
        "# ScanDar\n"
        "A phone photo in, a clean scan out. **Pick what to run first** — the six "
        "options are not interchangeable, and the same photo run through the wrong "
        "one just produces a confusing picture rather than an error."
    )

    with gr.Row():
        with gr.Column(scale=1):
            task = gr.Dropdown(
                choices=list(TASKS.keys()), value=DEFAULT_TASK, label="Model / task"
            )
            help_text = gr.Markdown(describe(DEFAULT_TASK))
            photo_in = gr.Image(label="Photo", type="numpy", sources=["upload", "clipboard"])
            if EXAMPLE_PHOTOS:
                gr.Examples(examples=[[str(p)] for p in EXAMPLE_PHOTOS], inputs=photo_in)
            run_button = gr.Button("Run", variant="primary")
            status_out = gr.Markdown()

        with gr.Column(scale=2):
            corners_out = gr.Image(label="Detected corners", visible=False)
            rectified_out = gr.Image(label="Rectified page", visible=False)
            scan_out = gr.Image(label="Result", visible=False)
            download_out = gr.File(label="Download result")

    task.change(fn=describe, inputs=task, outputs=help_text)
    run_button.click(
        fn=run,
        inputs=[task, photo_in],
        outputs=[corners_out, rectified_out, scan_out, status_out, download_out],
    )

if __name__ == "__main__":
    demo.launch()
