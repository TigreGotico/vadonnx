"""Model conversion / publishing helpers (maintainer-only).

These modules fetch upstream VAD models, verify (or export) their ONNX form, write a
signature sidecar + model card, and optionally publish to the ``TigreGotico``
HuggingFace org. Heavy dependencies (torch, nemo_toolkit) are imported lazily inside
the converters that need them, so importing this package never pulls them in.

Install with the ``[convert]`` (or ``[convert-marblenet]``) extra.
"""
