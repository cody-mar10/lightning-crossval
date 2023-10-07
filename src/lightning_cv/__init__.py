import logging

from . import callbacks, config, data, module, split, trainer, tuning, typehints, utils
from .__metadata__ import (
    __author__,
    __description__,
    __license__,
    __title__,
    __version__,
)
from .config import CrossValidationTrainerConfig
from .data import CrossValidationDataModule
from .module import BaseModelConfig, CrossValModule, CrossValModuleMixin
from .split import BaseCrossValidator, BaseGroupCrossValidator
from .trainer import CrossValidationTrainer

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_logger.addHandler(logging.StreamHandler())
