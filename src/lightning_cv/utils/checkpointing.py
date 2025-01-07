import dataclasses
from collections import defaultdict
from collections.abc import Mapping
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from inspect import Signature, getfile, signature
from pathlib import Path
from typing import Any, Optional, TypedDict, Union
from warnings import warn

import attrs
from lightning import Fabric, LightningModule
from lightning.fabric.wrappers import _unwrap_objects as fabric_unwrap
from torch.optim import Optimizer

from lightning_cv._optional import is_pydantic_basemodel
from lightning_cv.config import FoldState
from lightning_cv.typehints import FilePath, KwargType, MetricType
from lightning_cv.utils.serde import config_deserializer, config_serializer

MODEL_CLASS_KEY = "MODEL_CLASS"
OPTIMIZER_CLASS_KEY = "OPTIMIZER_CLASS"
MODEL_INIT_DESERIALIZABLE_KEY = "MODEL_INIT_DESERIALIZABLE"
MODEL_INIT_ARGS_KEY = "MODEL_INIT_ARGS"
MODEL_STATE_DICT = "model"
OPTIMIZER_STATE_DICT = "optimizer"
SCHEDULER_CLASS_KEY = "SCHEDULER_CLASS"
SCHEDULER_STATE_DICT = "scheduler"

_MISSING_SENTINEL = object()


class _ClassImportMetadata(TypedDict):
    module: str
    classname: str
    module_path: str


def _get_class_import_path(obj) -> _ClassImportMetadata:
    module = obj.__module__
    if isinstance(obj, type):  # passed class
        classname = obj.__name__
        module_path = getfile(obj)
    else:  # passed instance
        classname = obj.__class__.__name__
        module_path = getfile(obj.__class__)

    return _ClassImportMetadata(
        module=module, classname=classname, module_path=module_path
    )


def _is_deserializable(T: type) -> bool:
    return attrs.has(T) or dataclasses.is_dataclass(T) or is_pydantic_basemodel(T)


def _model_init_signature(model: type[LightningModule]) -> Signature:
    return signature(model)


def _check_model_init_signature_for_deserialization(
    sig: Signature,
) -> list[tuple[str, _ClassImportMetadata]]:
    deserializable_init_args: list[tuple[str, _ClassImportMetadata]] = list()

    for key, param in sig.parameters.items():
        if param.annotation is not param.empty and _is_deserializable(param.annotation):
            class_path = _get_class_import_path(param.annotation)
            deserializable_init_args.append((key, class_path))

    return deserializable_init_args


class CheckpointLoaderMixin:
    current_epoch: int
    global_step_per_fold: dict[int, int]
    is_first_epoch: bool
    current_val_metrics: MetricType
    checkpoint_dir: Path
    fabric: Fabric

    @staticmethod
    def _try_load_module_from_filepath(module_name: str, module_path: str):
        # this is the recipe from python stdlib docs
        import sys

        spec = spec_from_file_location(module_name, module_path)
        module = module_from_spec(spec)  # type: ignore
        sys.modules[module_name] = module
        spec.loader.exec_module(module)  # type: ignore
        return module

    @staticmethod
    def _try_load_module(module_name: str, module_path: str):
        try:
            return import_module(module_name)
        except ModuleNotFoundError:
            try:
                return CheckpointLoaderMixin._try_load_module_from_filepath(
                    module_name, module_path
                )
            except Exception as e:
                raise ImportError(f"Could not import module {module_path}") from e

    @staticmethod
    def _try_import_class(import_metadata: _ClassImportMetadata):
        module = CheckpointLoaderMixin._try_load_module(
            import_metadata["module"], import_metadata["module_path"]
        )
        try:
            return getattr(module, import_metadata["classname"])
        except AttributeError as e:
            raise ImportError(
                f"Could not import class {import_metadata['classname']} from module {module}: {import_metadata}"
            ) from e

    def _set_self_state(self, state_dict: KwargType):
        self.current_fold = state_dict.pop("current_fold")
        self.current_epoch = state_dict.pop("current_epoch")
        self.global_step_per_fold[self.current_fold] = state_dict.pop("global_step")

    def _load_model(self, state_dict: KwargType) -> LightningModule:
        model_class_path = state_dict.pop(MODEL_CLASS_KEY)
        model_class = self._try_import_class(model_class_path)

        if not issubclass(model_class, LightningModule):
            raise ValueError(
                f"Model class {model_class} is not a subclass of LightningModule"
            )

        model_init_args: KwargType = state_dict.pop(MODEL_INIT_ARGS_KEY)

        # need to deserialize any deserializable init args
        deserializable_init_args = state_dict.pop(MODEL_INIT_DESERIALIZABLE_KEY)

        for init_arg, init_arg_class_path in deserializable_init_args:
            init_arg_class = self._try_import_class(init_arg_class_path)
            config = model_init_args.pop(init_arg)
            model_init_args[init_arg] = config_deserializer(config, init_arg_class)

        # then build model
        model = model_class(**model_init_args)
        model.load_state_dict(state_dict.pop(MODEL_STATE_DICT))

        return model

    def _load_optimizer(
        self, state_dict: KwargType, model: LightningModule
    ) -> Optimizer:
        optimizer_class_path = state_dict.pop(OPTIMIZER_CLASS_KEY)
        optimizer_class = self._try_import_class(optimizer_class_path)

        optimizer = optimizer_class(
            model.parameters()
        )  # TODO: need to capture init args from state_dict
        optimizer.load_state_dict(state_dict.pop(OPTIMIZER_STATE_DICT))

        return optimizer

    def _load_scheduler(self, state_dict: KwargType, optimizer: Optimizer):
        scheduler_class_path = state_dict.pop(SCHEDULER_CLASS_KEY)
        scheduler_class = self._try_import_class(scheduler_class_path)

        scheduler = scheduler_class(
            optimizer
        )  # TODO: need to capture init args from state_dict
        scheduler.load_state_dict(state_dict.pop(SCHEDULER_STATE_DICT))

        return scheduler

    def load(self, checkpoint_path: FilePath) -> FoldState:
        warn(
            "Support for loading checkpoints is beta and may not work as expected",
            UserWarning,
        )
        state_dict = self.fabric.load(checkpoint_path)

        self._set_self_state(state_dict)
        model = self._load_model(state_dict)
        optimizer = self._load_optimizer(state_dict, model)
        if state_dict[SCHEDULER_STATE_DICT] is not None:
            scheduler = self._load_scheduler(state_dict, optimizer)
        else:
            scheduler = None

        return FoldState(model=model, optimizer=optimizer, scheduler=scheduler)

    def load_all_folds(
        self, checkpoint_dir: Path, ext: str = "ckpt"
    ) -> dict[int, FoldState]:
        paths = list(checkpoint_dir.glob(f"*.{ext}"))
        ckpt_status: defaultdict[str, list[Path]] = defaultdict(list)
        for path in paths:
            name = path.name
            if "training_complete" in name:
                ckpt_status["timeout"].append(path)
            elif "last" in name:
                ckpt_status["last"].append(path)
            else:
                ckpt_status["default"].append(path)

        fold_manager: dict[int, FoldState] = dict()
        if ckpt_status["timeout"]:
            for path in ckpt_status["timeout"]:
                # TODO: load checkpoints AND finish remaining checkpoints
                pass

        for path in paths:
            # this should set self.current_fold
            state = self.load(path)
            fold_manager[self.current_fold] = state
        return fold_manager


class CheckpointSaverMixin:
    current_epoch: int
    global_step_per_fold: dict[int, int]
    is_first_epoch: bool
    current_val_metrics: MetricType
    checkpoint_dir: Path
    fabric: Fabric

    def _unwrap_fabric_objects_in_place(self, state: FoldState):
        for key in ["model", "optimizer"]:
            setattr(state, key, fabric_unwrap(getattr(state, key)))

    def _checkpoint_name(
        self,
        fold: int,
        suffix: Optional[str] = None,
        include_val_loss: bool = True,
        ext: str = "ckpt",
        **name_kwargs,
    ) -> Path:
        # use new dict for consistent key order
        _name_kwargs = dict()
        _name_kwargs["epoch"] = self.current_epoch
        _name_kwargs["fold"] = fold

        if include_val_loss and not self.is_first_epoch:
            _name_kwargs["loss"] = f"{self.current_val_metrics['loss']:.4f}"
        else:
            _name_kwargs.update(name_kwargs)

        if suffix is not None:
            _name_kwargs["suffix"] = suffix

        _name = "_".join(
            f"{k}={v}" if k != "suffix" else str(v) for k, v in _name_kwargs.items()
        )
        return self.checkpoint_dir / f"{_name}.{ext}"

    def _serialize_state_for_saving(self, state: FoldState, fold: int) -> KwargType:
        self._unwrap_fabric_objects_in_place(state)

        ser_state = state.to_dict()
        ser_state.update(
            global_step=self.global_step_per_fold[fold],
            current_epoch=self.current_epoch,
            current_fold=fold,
        )

        # need to keep track of the optimizer and model classes
        # NOTE: this is required since lightning.Fabric does not allow module and optimizer class
        # types to be saved, even though this would probably be possible with pickling
        ser_state[MODEL_CLASS_KEY] = _get_class_import_path(state.model)
        ser_state[OPTIMIZER_CLASS_KEY] = _get_class_import_path(state.optimizer)
        if state.scheduler is not None:
            warn("Saving schedulers is beta and may not work as expected", UserWarning)
            ser_state[SCHEDULER_CLASS_KEY] = _get_class_import_path(
                state.scheduler["scheduler"]
            )

        # also add the init args for the model when reloading
        model_init_sig = _model_init_signature(state.model.__class__)  # type: ignore[arg-type]

        # NOTE: requires users to set all init args to the correspondingly named keys on the model
        model_init_args: KwargType = dict()
        for key in model_init_sig.parameters.keys():
            value = state.model.hparams.get(key, _MISSING_SENTINEL)

            if value is _MISSING_SENTINEL:
                value = getattr(state.model, key, _MISSING_SENTINEL)

            if value is _MISSING_SENTINEL:
                raise AttributeError(
                    f"Could not find attribute {key} in model {state.model}. This attribute "
                    "must be set as an instance attribute (ie self.key = value) in the model OR "
                    "be stored in the LightningModule's hparams dict (as a result of using "
                    "the LightningModule.save_hyperparameters method)."
                )

            model_init_args[key] = value

        ser_state[MODEL_INIT_ARGS_KEY] = config_serializer(model_init_args)

        # check if any init args are deserializable from dicts
        deserializable_model_init_args = (
            _check_model_init_signature_for_deserialization(model_init_sig)
        )

        ser_state[MODEL_INIT_DESERIALIZABLE_KEY] = deserializable_model_init_args

        return ser_state

    def _save(
        self,
        state: FoldState,
        fold: int,
        suffix: Optional[str] = None,
        ext: str = "ckpt",
        include_val_loss: bool = True,
        **name_kwargs: Any,
    ):
        ser_state = self._serialize_state_for_saving(state, fold)

        output = self._checkpoint_name(
            fold=fold,
            suffix=suffix,
            include_val_loss=include_val_loss,
            ext=ext,
            **name_kwargs,
        )

        self.fabric.save(output, ser_state)

    def save(
        self,
        states: Union[FoldState, Mapping[int, FoldState]],
        suffix: Optional[str] = None,
        ext: str = "ckpt",
        include_val_loss: bool = False,
        **name_kwargs,
    ):
        """Saves a checkpoint to the ``checkpoint_dir``

        Args:
            state: A mapping containing model, optimizer and lr scheduler.
        """
        if isinstance(states, Mapping):
            for fold, state in states.items():
                self._save(
                    state=state,
                    fold=fold,
                    suffix=suffix,
                    ext=ext,
                    include_val_loss=include_val_loss,
                    **name_kwargs,
                )
        else:
            self._save(
                state=states,
                suffix=suffix,
                ext=ext,
                include_val_loss=include_val_loss,
                **name_kwargs,
            )


class CurrentCheckpointsMixin:
    checkpoint_dir: Path

    def _keep_latest_checkpoints(self, paths: list[Path]) -> list[Path]:
        import re
        from operator import itemgetter

        pattern = re.compile(r"epoch=(\d+)_fold=(\d+)")

        path2ef = {
            path: tuple(int(x) for x in match.groups())
            for path in paths
            if (match := pattern.search(path.name))
        }
        latest_epoch = max(path2ef.values(), key=itemgetter(0))[0]
        latest_path2ef = {
            path: (epoch, fold)
            for path, (epoch, fold) in path2ef.items()
            if epoch == latest_epoch
        }

        return sorted(latest_path2ef, key=latest_path2ef.__getitem__)

    @staticmethod
    def get_latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
        """Returns the latest checkpoint from the ``checkpoint_dir``

        Args:
            checkpoint_dir: the directory to search for checkpoints
        """
        import os

        if not ckpt_dir.is_dir():
            return None

        # TODO: I don't think this works as expected
        items = sorted(os.listdir(ckpt_dir))

        if not items:
            return None

        return ckpt_dir.joinpath(items[-1])

    def current_ckptfiles(self) -> list[Path]:
        return list(self.checkpoint_dir.glob("*.ckpt"))


class CheckpointingMixin(
    CheckpointSaverMixin, CheckpointLoaderMixin, CurrentCheckpointsMixin
):
    pass
