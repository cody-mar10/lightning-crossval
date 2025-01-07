import pytest

from lightning_cv.utils.stopping import StopReasons


@pytest.mark.parametrize(
    ("reason", "expected_prefix"),
    [
        (StopReasons.TIMEOUT, "TIMEOUT"),
        (StopReasons.NO_IMPROVEMENT, "EARLY STOP"),
        (StopReasons.PRUNED_TRIAL, "PRUNED"),
        (StopReasons.LOSS_NAN_OR_INF, "NAN/INF"),
        (StopReasons.PERFORMANCE_STALLED, "STALLED"),
        (StopReasons.NULL, "COMPLETE"),
    ],
)
def test_stop_reason_message(reason: StopReasons, expected_prefix: str):
    msg = reason.message()
    assert msg.startswith(expected_prefix)
