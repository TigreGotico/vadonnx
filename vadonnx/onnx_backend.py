"""Generic ONNX VAD engine driven entirely by an :class:`~vadonnx.signature.IOSignature`.

Most supported models (Silero, TEN, FSMN, MarbleNet) differ only in their signature,
not their code, so they all run through this single class. A backend subclass is only
needed when a model requires glue that cannot be expressed declaratively.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from . import features as _features
from .base import VADModel
from .signature import IOSignature


class OnnxVAD(VADModel):
    """Run any declaratively-described VAD ONNX model frame by frame."""

    def __init__(
        self,
        model_path: str,
        signature: IOSignature,
        *,
        providers: Optional[List[str]] = None,
        intra_threads: int = 1,
        inter_threads: int = 1,
        threshold: float = 0.5,
        neg_threshold: Optional[float] = None,
    ):
        import onnxruntime as ort

        self.signature = signature
        self.model_path = model_path

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_threads
        opts.inter_op_num_threads = inter_threads
        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=providers or ["CPUExecutionProvider"],
        )
        self._model_inputs = {i.name for i in self.session.get_inputs()}
        self._output_names = [o.name for o in self.session.get_outputs()]

        super().__init__(
            sample_rate=signature.sample_rate,
            frame_size=signature.frame_size,
            stateful=signature.stateful,
            threshold=threshold,
            neg_threshold=neg_threshold,
        )

    # ------------------------------------------------------------------ #
    def _reset_state(self) -> None:
        self._state = {
            name: np.zeros(shape, dtype=np.float32)
            for name, shape in self.signature.state_inputs.items()
        }
        self._context = np.zeros(self.signature.context_size, dtype=np.float32)

    def _shape_audio(self, arr: np.ndarray) -> np.ndarray:
        layout = self.signature.audio_layout
        arr = np.asarray(arr, dtype=np.float32)
        if layout == "BT":
            return arr.reshape(1, -1)
        if layout == "T":
            return arr.reshape(-1)
        if layout == "BFT":  # (1, n_feat, n_frames)
            return np.ascontiguousarray(arr.T[None], dtype=np.float32)
        if layout == "BTF":  # (1, n_frames, n_feat)
            return np.ascontiguousarray(arr[None], dtype=np.float32)
        raise ValueError(f"unknown audio_layout {layout!r}")

    @staticmethod
    def _const(kind: str, value) -> np.ndarray:
        if kind in ("int64_scalar", "int64"):
            return np.array(value, dtype=np.int64)
        if kind in ("float_scalar", "float"):
            return np.array(value, dtype=np.float32)
        raise ValueError(f"unknown extra-input kind {kind!r}")

    def _reduce(self, arr) -> float:
        a = np.asarray(arr, dtype=np.float32)
        if self.signature.multiclass_collapse is not None:
            a = np.take(a, self.signature.multiclass_collapse, axis=-1).sum(axis=-1)
        mode = self.signature.prob_extract
        flat = a.reshape(-1)
        if flat.size == 0:
            return 0.0
        if mode == "scalar":
            return float(flat[0])
        if mode == "last":
            return float(flat[-1])
        if mode == "mean":
            return float(flat.mean())
        if mode.startswith("index:"):
            n = int(mode.split(":", 1)[1])
            return float(np.take(a, n, axis=-1).reshape(-1).mean())
        if mode.startswith("1-minus:"):
            n = int(mode.split(":", 1)[1])
            return float(1.0 - np.take(a, n, axis=-1).reshape(-1).mean())
        raise ValueError(f"unknown prob_extract {mode!r}")

    # ------------------------------------------------------------------ #
    def _infer_frame(self, frame_f32: np.ndarray) -> float:
        sig = self.signature
        if sig.context_size:
            frame_f32 = np.concatenate([self._context, frame_f32])
            self._context = frame_f32[-sig.context_size:].copy()
        if sig.feature:
            data = _features.extract(
                sig.feature, frame_f32, self.sample_rate, **sig.feature_params
            )
        else:
            data = frame_f32

        feed = {sig.audio_input: self._shape_audio(data)}
        for name, val in self._state.items():
            feed[name] = val
        for name, (kind, value) in sig.extra_inputs.items():
            feed[name] = self._const(kind, value)

        outs = self.session.run(None, feed)
        named = dict(zip(self._output_names, outs))

        for in_name, out_name in sig.state_output_map.items():
            self._state[in_name] = named[out_name]

        if isinstance(sig.prob_output, str):
            prob_arr = named[sig.prob_output]
        else:
            prob_arr = outs[sig.prob_output]
        return self._reduce(prob_arr)
