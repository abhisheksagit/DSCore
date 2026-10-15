"""Image embedders.

Two implementations behind one interface:

* `ClassicalEmbedder` needs no model weights. It is colour, texture and coarse
  layout. It is genuinely good at near-duplicates and same-garment-different-
  crop, and weak at generalising across angle and lighting.
* `SiglipOnnxEmbedder` is the production semantic encoder. SigLIP leads
  product-retrieval benchmarks, which is why it is the choice over CLIP.

The split matters operationally: the pipeline is useful and testable with the
classical embedder alone, so nothing blocks on getting weights onto the VPS,
and swapping in SigLIP is a config change rather than a rewrite.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from PIL import Image

from . import fingerprint as fp


def l2_normalise(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return (vec / norm).astype(np.float32) if norm else vec.astype(np.float32)


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, image: Image.Image) -> np.ndarray:
        """Return an L2-normalised float32 vector, so cosine == dot product."""


class ClassicalEmbedder:
    """Colour histogram + edge orientation + coarse luminance layout.

    Component weights set how much each signal can move a cosine score. Colour
    dominates deliberately: with one colourway per product it is Panache's
    single most discriminating cheap feature.
    """

    name = "classical-v1"

    def __init__(
        self,
        colour_weight: float = 1.0,
        edge_weight: float = 0.6,
        layout_weight: float = 0.4,
    ) -> None:
        self.colour_weight = colour_weight
        self.edge_weight = edge_weight
        self.layout_weight = layout_weight
        self.dim = len(self.embed(Image.new("RGB", (64, 64), (128, 128, 128))))

    def embed(self, image: Image.Image) -> np.ndarray:
        colour = fp.hsv_histogram(image)
        edges = fp.edge_orientation_histogram(image)
        layout = fp.luma_grid(image)
        vec = np.concatenate(
            [
                l2_normalise(colour) * self.colour_weight,
                l2_normalise(edges) * self.edge_weight,
                l2_normalise(layout) * self.layout_weight,
            ]
        )
        return l2_normalise(vec)


class SiglipOnnxEmbedder:
    """SigLIP vision tower exported to ONNX, run on CPU via onnxruntime.

    Export once (on a machine with Hugging Face access) and ship the .onnx to
    the VPS; onnxruntime keeps the runtime dependency small, with no torch.
    """

    name = "siglip-onnx"

    def __init__(
        self,
        onnx_path: str | Path,
        side: int = 384,
        mean: float = 0.5,
        std: float = 0.5,
        threads: int = 2,
    ) -> None:
        path = Path(onnx_path)
        if not path.exists():
            raise FileNotFoundError(
                f"SigLIP ONNX weights not found at {path}.\n"
                "Export them where huggingface.co is reachable:\n"
                "  optimum-cli export onnx --model google/siglip-base-patch16-384 "
                "--task feature-extraction siglip-onnx/\n"
                "then copy siglip-onnx/model.onnx to the VPS and point "
                "SIGLIP_ONNX_PATH at it."
            )
        import onnxruntime as ort  # imported lazily: optional dependency

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self._session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self.side = side
        self.mean = mean
        self.std = std
        self.dim = int(self._session.get_outputs()[0].shape[-1])

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        resized = image.convert("RGB").resize((self.side, self.side), Image.BICUBIC)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        return arr.transpose(2, 0, 1)[None, ...]

    def embed(self, image: Image.Image) -> np.ndarray:
        outputs = self._session.run(None, {self._input_name: self._preprocess(image)})
        vec = np.asarray(outputs[0]).reshape(-1)
        return l2_normalise(vec)


def load_embedder(onnx_path: str | None = None) -> Embedder:
    """Prefer SigLIP; fall back to classical with the reason made explicit.

    The caller is expected to surface `embedder.name` in the audit log, so a
    silent downgrade can never be mistaken for production accuracy.
    """
    if onnx_path:
        return SiglipOnnxEmbedder(onnx_path)
    return ClassicalEmbedder()
