"""Optional backend subclasses for models that need glue beyond a plain signature."""
from .fsmn import FsmnVAD
from .marblenet import MarbleVAD
from .pyannote import PyannoteVAD
from .speechbrain import SbVAD
from .ten import TenVAD

__all__ = ["TenVAD", "FsmnVAD", "MarbleVAD", "SbVAD", "PyannoteVAD"]
