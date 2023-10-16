import logging

from lightning_cv import (
    callbacks,
    config,
    data,
    module,
    split,
    trainer,
    tuning,
    typehints,
    utils,
)
from lightning_cv.__metadata__ import (
    __author__,
    __description__,
    __license__,
    __title__,
    __version__,
)
from lightning_cv.config import CrossValidationTrainerConfig
from lightning_cv.data import CrossValidationDataModule
from lightning_cv.module import BaseModelConfig, CrossValModule, CrossValModuleMixin
from lightning_cv.split import BaseCrossValidator, BaseGroupCrossValidator
from lightning_cv.trainer import CrossValidationTrainer

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_logger.addHandler(logging.StreamHandler())
