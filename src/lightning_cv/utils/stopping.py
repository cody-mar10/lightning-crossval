from enum import Enum, auto


class StopReasons(Enum):
    TIMEOUT = auto()
    NO_IMPROVEMENT = auto()
    PRUNED_TRIAL = auto()
    LOSS_NAN_OR_INF = auto()
    PERFORMANCE_STALLED = auto()
    NULL = auto()

    def message(self) -> str:
        if self == StopReasons.NULL:
            return "COMPLETE: Training {max_epochs} epochs completed."

        if self == StopReasons.NO_IMPROVEMENT:
            return (
                "EARLY STOP: Training stopped early due to no improvement in {monitor}"
            )

        if self == StopReasons.TIMEOUT:
            return "TIMEOUT: Training stopped early due to running out of time."

        if self == StopReasons.LOSS_NAN_OR_INF:
            return "NAN/INF: Training stopped early due to NaN or Inf loss."

        if self == StopReasons.PERFORMANCE_STALLED:
            return (
                "STALLED: Training stopped early due to performance stalling. {monitor} is not "
                "improving during the monitored stage."
            )

        return (
            "PRUNED: Training stopped early due to bad trial pruning during hyperparameter "
            "tuning."
        )
