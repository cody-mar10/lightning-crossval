import logging

from lightning_cv.__metadata__ import (
    __author__,
    __description__,
    __license__,
    __title__,
    __version__,
)
from lightning_cv.config import CrossValidationTrainerConfig
from lightning_cv.data import CrossValidationDataModule, CrossValidationDataModuleMixin
from lightning_cv.split import BaseCrossValidator, BaseGroupCrossValidator
from lightning_cv.trainer import CrossValidationTrainer

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_logger.addHandler(logging.StreamHandler())

__all__ = [
    "CrossValidationDataModule",
    "CrossValidationDataModuleMixin",
    "BaseCrossValidator",
    "BaseGroupCrossValidator",
    "CrossValidationTrainer",
    "CrossValidationTrainerConfig",
    "__author__",
    "__description__",
    "__license__",
    "__title__",
    "__version__",
]
