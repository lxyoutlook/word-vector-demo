"""Resolve the original two-file Gensim model without duplicating it."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILENAME = "Tencent_AILab_ChineseEmbedding.bin"


def model_path() -> Path:
    override = os.environ.get("WORD_VECTOR_MODEL_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return ROOT / "data" / FILENAME


def model_files(path: Path) -> tuple[Path, Path]:
    return path, Path(str(path) + ".vectors.npy")


def model_identity(path: Path) -> tuple:
    """Invalidate caches if either file is replaced, including only the array."""
    identity = [str(path.resolve())]
    for file in model_files(path):
        stat = file.stat()
        identity.extend((stat.st_size, stat.st_mtime_ns))
    return tuple(identity)
