from typing import TYPE_CHECKING, Any, Literal, Union

from lightning import Fabric as LightningFabric
from lightning_utilities.core.rank_zero import rank_zero_warn

if TYPE_CHECKING:
    from lightning.fabric.accelerators.accelerator import Accelerator
    from lightning.fabric.loggers.logger import Logger
    from lightning.fabric.plugins import CheckpointIO, ClusterEnvironment, Precision
    from lightning.fabric.strategies.strategy import Strategy
    from torch.nn import Module
    from torch.optim import Optimizer

from lightning import Callback as LightingCallback
from lightning import LightningModule

from lightning_cv.callbacks.base import Callback as LCVCallback
from lightning_cv.callbacks.base import CallbackT

PrecisionT = Union[
    None,
    Literal[
        64,
        32,
        16,
        "transformer-engine",
        "transformer-engine-float16",
        "16-true",
        "16-mixed",
        "bf16-true",
        "bf16-mixed",
        "32-true",
        "64-true",
        "64",
        "32",
        "16",
        "bf16",
    ],
]

_PluginsT = Union["Precision", "ClusterEnvironment", "CheckpointIO"]
PluginsT = Union[_PluginsT, list[_PluginsT], None]

AcceleratorT = Union[Literal["auto", "cpu", "gpu", "tpu", "cuda", "mps"], "Accelerator"]
StrategyT = Union[
    Literal["dp", "ddp", "ddp_spawn", "ddp_notebook", "fsdp", "deepspeed", "auto"],
    "Strategy",
]
DevicesT = Union[list[int], Literal["auto"], int]


LoggerT = Union["Logger", list["Logger"], None]


LightningFabric()


class Fabric(LightningFabric):

    _lcv_callbacks: list[LCVCallback]
    _model_callbacks: list[LightningModule]
    _lightning_callbacks: list[LightingCallback]

    def __init__(
        self,
        *,
        accelerator: AcceleratorT = "auto",
        strategy: StrategyT = "auto",
        devices: DevicesT = "auto",
        num_nodes: int = 1,
        precision: PrecisionT = None,
        plugins: PluginsT = None,
        callbacks: CallbackT = None,
        loggers: LoggerT = None,
    ) -> None:
        super().__init__(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            num_nodes=num_nodes,
            precision=precision,
            plugins=plugins,
            callbacks=callbacks,
            loggers=loggers,
        )
        # could probably merge lightning and model callbacks since they have the same callbacks
        # except I think they expect a lightning Trainer?
        self._lcv_callbacks = []
        self._model_callbacks = []
        self._lightning_callbacks = []
        # only move each callback once
        self._seen_callbacks = set()

        self._already_warned_about_callbacks = False

        self._move_all_callbacks()

    def _move_callback(self, callback):
        if callback in self._seen_callbacks:
            return

        if isinstance(callback, LCVCallback):
            self._lcv_callbacks.append(callback)
        elif isinstance(callback, LightningModule):
            self._model_callbacks.append(callback)
        elif isinstance(callback, LightingCallback):
            self._lightning_callbacks.append(callback)
        else:
            raise ValueError(f"Unknown callback type: {callback}")

        self._seen_callbacks.add(callback)

    def _move_all_callbacks(self):
        for callback in self._callbacks:
            self._move_callback(callback)

    def setup(
        self,
        module: "Module",
        *optimizers: "Optimizer",
        move_to_device: bool = True,
        _reapply_compile: bool = True,
    ) -> Any:
        output = super().setup(
            module,
            *optimizers,
            move_to_device=move_to_device,
            _reapply_compile=_reapply_compile,
        )

        self._move_all_callbacks()
        return output

    def _call(self, callback_list: list, hook_name: str, *args, **kwargs) -> None:
        # this is the implementation of lightning.Fabric.call
        # but allowing an input for the callback list
        for callback in callback_list:
            method = getattr(callback, hook_name, None)
            if method is None:
                continue

            if not callable(
                method
            ):  # pragma: no cover <- this is using lightning methods that should already be tested
                rank_zero_warn(
                    f"Callback {callback} does not have a callable method {hook_name}"
                )
                continue

            method(*args, **kwargs)

    def call(self, hook_name: str, *args, **kwargs) -> None:
        self._call(self._lcv_callbacks, hook_name, *args, **kwargs)

        # TODO: need to handle these callbacks differently since might have diff names, etc
        # and don't take a CVTrainer
        # self._call(self._model_callbacks, hook_name, *args, **kwargs)
        # self._call(self._lightning_callbacks, hook_name, *args, **kwargs)
        if self._model_callbacks and not self._already_warned_about_callbacks:
            rank_zero_warn(
                "Model callbacks are not yet supported in Fabric. "
                "Please use Lightning callbacks instead."
            )
            self._already_warned_about_callbacks = True

        if self._lightning_callbacks and not self._already_warned_about_callbacks:
            rank_zero_warn(
                "Lightning callbacks are not yet supported in Fabric. "
                "Please use Lightning callbacks instead."
            )
            self._already_warned_about_callbacks = True
