from lightning.fabric.strategies.ddp import DDPStrategy
from lightning.fabric.strategies.deepspeed import DeepSpeedStrategy
from lightning.fabric.strategies.fsdp import FSDPStrategy
from lightning.fabric.strategies.strategy import Strategy
from lightning.fabric.strategies.xla import XLAStrategy

_DISTRIBUTED_STRATEGIES = (DDPStrategy, DeepSpeedStrategy, FSDPStrategy, XLAStrategy)


def is_distributed(strategy: Strategy) -> bool:
    if hasattr(strategy, "is_distributed"):  # pragma: no cover
        return strategy.is_distributed  # type: ignore

    return isinstance(strategy, _DISTRIBUTED_STRATEGIES)
