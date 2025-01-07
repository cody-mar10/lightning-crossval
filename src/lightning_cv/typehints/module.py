from typing import TYPE_CHECKING, Any, Mapping, Union

if TYPE_CHECKING:
    from torch import Tensor
    from torch.optim import Optimizer

ConfigOptimizerOutput = Union["Optimizer", Mapping]
ForwardOutputType = Union["Tensor", Mapping[str, Any]]
