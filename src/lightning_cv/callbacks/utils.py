from collections.abc import Sequence
from pathlib import Path

from lightning_cv.callbacks.base import CallbackT, _CallbackT


def standardize_callbacks(callbacks: CallbackT) -> list[_CallbackT]:
    """Standardize all possible callbacks when creating a `CrossValidationTrainer` or
    `LightningFabric` instance into a list."""
    if callbacks is None:
        return []

    if isinstance(callbacks, Sequence):
        return list(callbacks)

    return [callbacks]


def remove_old_checkpoints(checkpoint_dir: Path, pattern: str = "*.ckpt"):
    """Remove old checkpoints from a directory based on a globbing pattern.

    Args:
        checkpoint_dir (Path): checkpointing directory
        pattern (Optional[str], optional): glob pattern for checkpoint files to remove. Defaults to '*.ckpt'.
    """
    for file in checkpoint_dir.glob(pattern):
        file.unlink()
