# TODO: test functionality of all callbacks
# some of these may need to be integration tests?

import typing
from datetime import timedelta
from pathlib import Path

import lightning as L
import pytest
import torch
from conftest import _add_callback
from torch.optim import Optimizer

from lightning_cv.callbacks.base import Callback
from lightning_cv.callbacks.timer import Intervals
from lightning_cv.data import CrossValidationDataModule
from lightning_cv.trainer import CrossValidationTrainer, TrainerState  # noqa
from lightning_cv.typehints import KwargType, MetricType
from lightning_cv.utils.stopping import StopReasons


def dir_is_empty(dir: Path):
    if not dir.is_dir() or not dir.exists():
        raise ValueError(f"{dir} is not a directory or does not exist")
    return not any(dir.iterdir())


class TestModelModeCallback:
    from lightning_cv.callbacks.mode import EvalMode, TrainMode

    # TODO: need an integration test to show that this happens at the expected time...
    def test_train_mode(
        self, cv_trainer_with_cleanup: CrossValidationTrainer, model: L.LightningModule
    ):
        model.eval()

        assert not model.training

        callback = self.TrainMode()
        callback.on_train_epoch_start_per_fold(cv_trainer_with_cleanup, model, 10, 1, 1)

        assert model.training

    def test_eval_mode(
        self, cv_trainer_with_cleanup: CrossValidationTrainer, model: L.LightningModule
    ):
        model.train()

        assert model.training

        callback = self.EvalMode()
        callback.on_validation_start_per_fold(cv_trainer_with_cleanup, model, 10)

        assert not model.training


class TestTimerCallback:
    from lightning_cv.callbacks.timer import Timer

    DURATION = timedelta(milliseconds=100)

    @pytest.mark.parametrize("training_only", [True, False])
    @pytest.mark.parametrize("interval", typing.get_args(Intervals))
    def test_trainer_status(
        self,
        cv_trainer_with_cleanup: CrossValidationTrainer,
        slow_model_config,
        datamodule: CrossValidationDataModule,
        interval: Intervals,
        training_only: bool,
    ):
        callback = self.Timer(
            interval=interval,
            duration=self.DURATION,
            save_latest=False,
            training_only=training_only,
        )
        _add_callback(cv_trainer_with_cleanup, callback)

        cv_trainer_with_cleanup.train_with_cross_validation(
            datamodule=datamodule, model_config=slow_model_config
        )

        assert callback._stopped
        assert cv_trainer_with_cleanup.status == StopReasons.TIMEOUT

        if training_only and interval != "fold":
            # Here is how the trainer loop works and when the Timer checks if time to stop:
            # trainer.cross_val_loop:
            #     2. for each fold:
            #         a. train_loop <- trainer.state == TRAINING
            #             i. on_train_epoch_end_per_fold <- Timer stop?
            #             ii. on_train_batch_end_per_fold <- Timer stop?
            #         b. val_loop <- trainer.state == VALIDATING
            #     3. on_train_fold_end <- Timer stop?

            # so basically if we end on batch/epoch/immediately AND only check during training,
            # the trainer will have been training and not validating at that moment

            assert cv_trainer_with_cleanup.state_when_stopped == TrainerState.TRAINING
        else:
            # can stop during validation OR stopping at the end training each fold
            # if stopping after a complete set of folds, then the val_loops will alos be ran
            # so the trainer should've been validating when it stopped

            # data is so small that it should be validating
            assert cv_trainer_with_cleanup.state_when_stopped == TrainerState.VALIDATING

    @pytest.mark.parametrize("save_latest", [True, False])
    def test_checkpointing(
        self,
        cv_trainer_with_cleanup: CrossValidationTrainer,
        save_latest: bool,
        datamodule: CrossValidationDataModule,
        slow_model_config,
    ):
        callback = self.Timer(
            interval="epoch",
            duration=self.DURATION,
            save_latest=save_latest,
        )
        _add_callback(cv_trainer_with_cleanup, callback)

        cv_trainer_with_cleanup.train_with_cross_validation(
            datamodule=datamodule, model_config=slow_model_config
        )

        # key error max epoch?

        ckptfiles = [
            file
            for file in cv_trainer_with_cleanup.current_ckptfiles()
            # trainer will normally be saving checkpoints, so need to filter
            if "latest" in file.name
        ]

        if save_latest:
            assert len(ckptfiles) == cv_trainer_with_cleanup.num_folds
            assert all("latest" in ckpt.name for ckpt in ckptfiles)
        else:
            assert len(ckptfiles) == 0

    @pytest.mark.parametrize(
        ("time", "expected_seconds"),
        [
            ("01:21", 81.0),  # 1 minute 21 seconds
            ("01:21:30", 4890.0),  # 1 hour 21 minutes 30 seconds
            ("02:22:12:30", 252750.0),  # 2 days 22 hours 12 minutes 30 seconds
        ],
    )
    def test_str_times(self, time: str, expected_seconds: float):
        timer = self.Timer(duration=time)
        assert isinstance(timer._duration, float)
        assert timer._duration == expected_seconds

        timedelta_dict = self.Timer._str_to_timedelta_dict(time)

        assert isinstance(timedelta_dict, dict)
        td = self.Timer._dict_to_timedelta(timedelta_dict)
        assert td.total_seconds() == expected_seconds

    def test_str_times_with_error(self):
        with pytest.raises(ValueError):
            self.Timer(duration="01:21:30:30:30")

    @pytest.mark.parametrize(
        "buffer",
        [
            timedelta(seconds=20),
            {"seconds": 20},
        ],
    )
    def test_buffer(self, buffer):
        timer = self.Timer(duration=timedelta(minutes=1), buffer=buffer)
        assert timer._duration == 40.0
        assert "buffer" in timer._repr

    def test_no_duration(self):
        timer = self.Timer()
        assert timer._duration is None
        assert timer._is_time_up() is False


class TestLRMonitorCallback:
    from lightning_cv.callbacks.lr_monitor import LearningRateMonitor

    METRICS = {"loss": 0.123}

    # TODO: this should be tested with a variety of optimizers
    def test_model_optimizer_has_lr(self, model: L.LightningModule):
        optimizer: "Optimizer" = model.configure_optimizers()  # type: ignore
        assert len(optimizer.param_groups) == 1

        group = optimizer.param_groups[0]
        assert "lr" in group

    def test_lr_added_to_metrics(
        self, cv_trainer_with_cleanup: CrossValidationTrainer, model: L.LightningModule
    ):
        metrics = {**self.METRICS}
        assert metrics == self.METRICS

        optimizer: Optimizer = model.configure_optimizers()  # type: ignore
        lrm = self.LearningRateMonitor()
        lrm.on_before_log_metrics(cv_trainer_with_cleanup, metrics, optimizer)

        assert metrics == {**self.METRICS, "lr": model.LR}


@pytest.mark.parametrize(
    "callbacks",
    [
        Callback(),
        None,
        [Callback(), Callback()],
    ],
)
def test_standardize_callbacks(callbacks):
    from lightning_cv.callbacks.utils import standardize_callbacks

    new_callbacks = standardize_callbacks(callbacks)

    assert isinstance(new_callbacks, list)

    if callbacks is None:
        assert new_callbacks == []
    elif isinstance(callbacks, Callback):
        assert new_callbacks == [callbacks]
        assert new_callbacks[0] is callbacks
    elif isinstance(callbacks, list):
        assert new_callbacks == callbacks


class TestEarlyStoppingCallback:
    from lightning_cv.callbacks.stopping import EarlyStopping

    EXPECTED_MODES = {"min", "max"}

    def _setup_mode(self, mode: str):
        return self.EarlyStopping(mode=mode)  # type: ignore[arg-type]

    @pytest.mark.parametrize("mode", ["min", "max", "minimum", "maX", "INVALID_MODE"])
    def test_mode(self, mode: str):
        if mode in self.EXPECTED_MODES:
            callback = self._setup_mode(mode)
            assert callback.mode == mode
        else:
            with pytest.raises(RuntimeError):
                self._setup_mode(mode=mode)

    @pytest.mark.parametrize("mode", ["min", "max"])
    def test_comparison_op(self, mode: str):
        callback = self._setup_mode(mode)

        if mode == "min":
            expected_monitor_op = torch.lt
        else:
            expected_monitor_op = torch.gt

        assert callback.monitor_op is expected_monitor_op

    @pytest.mark.parametrize("mode", ["min", "max"])
    def test_initial_best_score(self, mode: str):
        callback = self._setup_mode(mode)

        if mode == "min":
            expected_best_score = torch.tensor(float("inf"))
        else:
            expected_best_score = torch.tensor(float("-inf"))

        assert callback.best_score == expected_best_score

    def test_stopping(
        self,
        long_cv_trainer_with_cleanup: CrossValidationTrainer,
        model_config,
        datamodule: CrossValidationDataModule,
    ):
        callback = self.EarlyStopping(patience=2, min_delta=0.25)
        _add_callback(long_cv_trainer_with_cleanup, callback)

        long_cv_trainer_with_cleanup.train_with_cross_validation(
            datamodule=datamodule, model_config=model_config
        )

        assert long_cv_trainer_with_cleanup.status == StopReasons.NO_IMPROVEMENT

    @pytest.fixture
    def metrics(self) -> MetricType:
        return {
            "loss": torch.tensor(0.5),
            "accuracy": torch.tensor(0.9),
            "inf_metric": torch.tensor(torch.inf),
            "improved_metric": torch.tensor(-1.0),
        }

    @pytest.mark.parametrize("metric", ["loss", "accuracy", "INVALID_METRIC"])
    def test_get_current_metric(self, metric: str, metrics: MetricType):
        callback = self.EarlyStopping(monitor=metric)
        if metric == "INVALID_METRIC":
            with pytest.raises(RuntimeError):
                callback._get_current_metric(metrics)
        else:
            assert callback._get_current_metric(metrics) == metrics[metric]

    @pytest.mark.parametrize(
        "metric", ["loss", "accuracy", "inf_metric", "improved_metric"]
    )
    def test_evaluate_stopping_criteria(self, metric: str, metrics: MetricType):
        kwargs = {
            "monitor": metric,
            "min_delta": 0.5,
            "patience": 1,
        }

        if metric == "accuracy":
            kwargs["mode"] = "max"
            best_score = 1.0
        else:
            kwargs["mode"] = "min"
            best_score = 0.0

        callback = self.EarlyStopping(**kwargs)
        # should not be possible to improve over this
        callback.best_score = torch.tensor(best_score)

        should_stop, reason = callback._evaluate_stopping_criteria(metrics[metric])

        # the only time reason is None is when there was no improvement but patience is > 1

        assert reason is not None

        if metric == "inf_metric":
            assert "not finite" in reason
            assert should_stop
        elif metric == "improved_metric":
            assert not should_stop
            assert "New best score" in reason
        else:
            # loss and accuracy did not improve so there is stopping
            assert should_stop
            assert "did not improve" in reason

    @pytest.mark.parametrize("metric", ["loss", "accuracy"])
    def test_stopping_threshold(self, metric: str, metrics: MetricType):
        kwargs: KwargType = {"monitor": metric}

        if metric == "accuracy":
            kwargs["mode"] = "max"
            best_score = 1.0
            kwargs["stopping_threshold"] = 0.85
        else:
            kwargs["mode"] = "min"
            best_score = 0.0
            kwargs["stopping_threshold"] = 0.55

        callback = self.EarlyStopping(**kwargs)
        # should not be possible to improve over this BUT hit the stopping threshold
        callback.best_score = torch.tensor(best_score)

        should_stop, reason = callback._evaluate_stopping_criteria(metrics[metric])

        assert should_stop
        assert reason is not None
        assert "Stopping threshold reached" in reason

    def test_divergence_threshold(self, metrics: MetricType):
        metric = "loss"
        kwargs = {"monitor": metric, "divergence_threshold": 0.5}

        callback = self.EarlyStopping(**kwargs)

        should_stop, reason = callback._evaluate_stopping_criteria(metrics[metric])

        assert should_stop
        assert reason is not None
        assert "Divergence threshold reached" in reason


class TestProgessBar:
    from lightning_cv.callbacks.progress import Tqdm

    @pytest.mark.parametrize(
        ("n", "expected"),
        [
            (1, "1"),
            (1.0, "1.000"),
            ("1", "1.000"),
            ("1.23e-3", "1.23e-3"),
            ("2.34", "2.340"),
            ("1.23456", "1.23456"),
        ],
    )
    def test_format_num(self, n: typing.Union[int, float, str], expected: str):
        formatted = self.Tqdm.format_num(n)
        assert isinstance(formatted, str)
        assert formatted == expected

        if isinstance(n, str) and "e" in n:
            assert formatted == n

        if not isinstance(n, int):
            assert len(formatted) >= 5  # this is the padding size

    @pytest.mark.parametrize("n", ["1.23.4", "ABC"])
    def test_invalid_format_num_input(self, n: str):
        formatted = self.Tqdm.format_num(n)
        assert isinstance(formatted, str)
        assert n == formatted


class TestValPerfPlateauMonitor:
    from lightning_cv.callbacks.perf_monitor import ValPerfPlateauMonitor

    @pytest.fixture
    def basic_monitor(self) -> ValPerfPlateauMonitor:
        monitor = self.ValPerfPlateauMonitor()
        monitor.metrics = [1.0, 2.0, 3.0]
        return monitor

    def test_mean(self, basic_monitor: ValPerfPlateauMonitor):
        assert basic_monitor.mean() == 2.0

    @pytest.mark.parametrize(
        ("xbar", "expected"), [(None, 1.0), (2.5, 1.1726039399558574)]
    )
    def test_std(
        self,
        basic_monitor: ValPerfPlateauMonitor,
        xbar: typing.Optional[float],
        expected: float,
    ):
        calc_std = basic_monitor.std(xbar=xbar)
        assert calc_std == expected

    def test_push_metric(self):
        monitor = self.ValPerfPlateauMonitor(monitor="loss")
        monitor.push_metric({"loss": torch.tensor(0.5)})
        assert monitor.metrics == [0.5]

    def test_clear_metrics(self, basic_monitor: ValPerfPlateauMonitor):
        assert len(basic_monitor.metrics) == 3
        basic_monitor.clear_metrics()
        assert len(basic_monitor.metrics) == 0

    @pytest.mark.parametrize(
        ("metrics", "expected"),
        [
            ([], False),
            ([1.0], False),
            ([1.0, 1.0, 1.0], True),
            ([1e-8, 1.23e-4, 1e-5], False),
            ([1e-8, 1e-8, 1e-8], False),
        ],
    )
    def test_should_stop(self, metrics: list[float], expected: bool):
        monitor = self.ValPerfPlateauMonitor()
        monitor.metrics = metrics
        assert monitor.should_stop() is expected

    def _check_trainer_report(
        self,
        monitor: ValPerfPlateauMonitor,
        trainer: CrossValidationTrainer,
        expected: StopReasons,
    ):
        assert trainer.status == expected

        should_stop = monitor.should_stop()
        assert trainer.should_stop is should_stop

    @pytest.mark.parametrize(
        ("metrics", "expected"),
        [
            ([], StopReasons.NULL),
            ([1.0], StopReasons.NULL),
            ([1.0, 1.0, 1.0], StopReasons.PERFORMANCE_STALLED),
            ([1e-8, 1.23e-4, 1e-5], StopReasons.NULL),
            ([1e-8, 1e-8, 1e-8], StopReasons.NULL),
        ],
    )
    def test_trainer_report(
        self,
        cv_trainer_with_cleanup: CrossValidationTrainer,
        metrics: list[float],
        expected: StopReasons,
    ):
        monitor = self.ValPerfPlateauMonitor()
        monitor.metrics = metrics
        monitor.report(cv_trainer_with_cleanup)

        self._check_trainer_report(monitor, cv_trainer_with_cleanup, expected)

    def test_on_validation_start_per_fold(
        self,
        cv_trainer_with_cleanup: CrossValidationTrainer,
        basic_monitor: ValPerfPlateauMonitor,
        model,
    ):
        assert len(basic_monitor.metrics) == 3

        _add_callback(cv_trainer_with_cleanup, basic_monitor)
        basic_monitor.on_validation_start_per_fold(cv_trainer_with_cleanup, model, 1)

        assert len(basic_monitor.metrics) == 0

    def test_on_validation_batch_end_per_fold(
        self, cv_trainer_with_cleanup: CrossValidationTrainer
    ):
        monitor = self.ValPerfPlateauMonitor(monitor="loss")
        metric = {"loss": torch.tensor(0.5)}
        monitor.on_validation_batch_end_per_fold(
            cv_trainer_with_cleanup, metric, None, 1
        )

        assert monitor.metrics == [0.5]

    @pytest.mark.parametrize(
        ("metrics", "expected"),
        [
            ([], StopReasons.NULL),
            ([1.0], StopReasons.NULL),
            ([1.0, 1.0, 1.0], StopReasons.PERFORMANCE_STALLED),
            ([1e-8, 1.23e-4, 1e-5], StopReasons.NULL),
            ([1e-8, 1e-8, 1e-8], StopReasons.NULL),
        ],
    )
    def test_on_validation_end_per_fold(
        self,
        cv_trainer_with_cleanup: CrossValidationTrainer,
        metrics: list[float],
        expected: StopReasons,
    ):
        monitor = self.ValPerfPlateauMonitor()
        monitor.metrics = metrics
        _add_callback(cv_trainer_with_cleanup, monitor)
        monitor.on_validation_end_per_fold(cv_trainer_with_cleanup)

        self._check_trainer_report(monitor, cv_trainer_with_cleanup, expected)

    # TODO: I think all "on_".... callback hooks need to be integration tests? <- they can be for real cases?


class TestModelCheckpoint:
    from lightning_cv.callbacks.checkpoint import ModelCheckpoint

    @pytest.fixture
    def current_val_metrics(self) -> MetricType:
        return {"loss": torch.tensor(0.5)}

    @pytest.fixture
    def basic_checkpoint(self) -> ModelCheckpoint:
        return self.ModelCheckpoint()

    @pytest.fixture
    def cv_trainer_with_setup(
        self, preset_cv_trainer_with_cleanup: CrossValidationTrainer
    ):
        preset_cv_trainer_with_cleanup.is_first_epoch = False

        # for this unit test, the trainer hasn't actually trained anything
        # so the overall checkpoint dir should look like this:
        # PYTEST_CHECKPOINT_DIR/lightning_logs
        # in real cases (integration test), it would look like this:
        # PYTEST_CHECKPOINT_DIR/lightning_logs/version_0/...
        # THUS, we need to make the dir here
        preset_cv_trainer_with_cleanup.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return preset_cv_trainer_with_cleanup

    def test_init_worst_of_best_epochs(self, basic_checkpoint: ModelCheckpoint):
        assert basic_checkpoint.worst_of_best_epochs() == -1

    @pytest.mark.parametrize("save_top_k", [0, 1, 2, -1])
    def test_save_top_k_ckpt(
        self, cv_trainer_with_setup: CrossValidationTrainer, save_top_k: int
    ):
        checkpoint = self.ModelCheckpoint(save_top_k=save_top_k)
        for epoch, val_loss in enumerate([0.5, 0.25, 0.125]):
            cv_trainer_with_setup.current_val_metrics = {"loss": torch.tensor(val_loss)}
            cv_trainer_with_setup.current_epoch = epoch

            checkpoint._save_topk_ckpt(cv_trainer_with_setup, val_loss)

        if save_top_k == 0:
            # no saving occurs
            assert dir_is_empty(cv_trainer_with_setup.checkpoint_dir)
        else:
            ckptfiles = cv_trainer_with_setup.current_ckptfiles()

            if save_top_k == 1:
                # 1 per fold
                assert len(ckptfiles) == len(cv_trainer_with_setup.fold_manager)
                assert all("0.125" in ckpt.name for ckpt in ckptfiles)
            elif save_top_k == 2:
                # 2 per fold
                assert (
                    len(ckptfiles)
                    == len(cv_trainer_with_setup.fold_manager) * save_top_k
                )

                assert all(
                    "0.125" in ckpt.name or "0.25" in ckpt.name for ckpt in ckptfiles
                )
            else:
                # everything gets saved
                assert len(ckptfiles) == cv_trainer_with_setup.num_folds * 3

    @pytest.mark.parametrize("metric", [float("inf"), float("-inf"), float("nan")])
    def test_nonfinite_save_top_k_ckpt(
        self,
        cv_trainer_with_setup: CrossValidationTrainer,
        basic_checkpoint: ModelCheckpoint,
        metric: float,
    ):
        basic_checkpoint._save_topk_ckpt(cv_trainer_with_setup, metric)

        assert dir_is_empty(cv_trainer_with_setup.checkpoint_dir)

    @pytest.mark.parametrize("current_epoch", [0, 5])
    def test_on_train_fold_end(
        self,
        cv_trainer_with_setup: CrossValidationTrainer,
        current_val_metrics: MetricType,
        current_epoch: int,
    ):
        checkpoint = self.ModelCheckpoint(every_n_epochs=2)  # 0 saves, 5 does not

        cv_trainer_with_setup.current_val_metrics = current_val_metrics
        cv_trainer_with_setup.current_epoch = current_epoch
        checkpoint.on_train_fold_end(cv_trainer_with_setup)

        if current_epoch == 0:
            ckptfiles = cv_trainer_with_setup.current_ckptfiles()
            assert len(ckptfiles) == cv_trainer_with_setup.num_folds

            loss_str = str(current_val_metrics["loss"].item())
            assert all(loss_str in ckpt.name for ckpt in ckptfiles)
        else:
            assert dir_is_empty(cv_trainer_with_setup.checkpoint_dir)

    def test_on_train_fold_end_without_current_val_metrics(
        self,
        cv_trainer_with_setup: CrossValidationTrainer,
        basic_checkpoint: ModelCheckpoint,
    ):
        basic_checkpoint.on_train_fold_end(cv_trainer_with_setup)
        assert dir_is_empty(cv_trainer_with_setup.checkpoint_dir)

    @pytest.mark.parametrize("save_last", [True, False])
    def test_on_train_end(
        self, cv_trainer_with_setup: CrossValidationTrainer, save_last: bool
    ):
        checkpoint = self.ModelCheckpoint(save_last=save_last)
        checkpoint.on_train_end(cv_trainer_with_setup)

        if save_last:
            ckptfiles = cv_trainer_with_setup.current_ckptfiles()
            assert len(ckptfiles) == cv_trainer_with_setup.num_folds
            assert all("last" in ckpt.name for ckpt in ckptfiles)
        else:
            assert dir_is_empty(cv_trainer_with_setup.checkpoint_dir)

    def test_tracking_best_epoch(self, cv_trainer_with_setup: CrossValidationTrainer):
        checkpoint = self.ModelCheckpoint(save_top_k=3)
        for epoch, val_loss in enumerate([0.5, 0.25, 0.125]):
            cv_trainer_with_setup.current_val_metrics = {"loss": torch.tensor(val_loss)}
            cv_trainer_with_setup.current_epoch = epoch
            checkpoint._save_topk_ckpt(cv_trainer_with_setup, val_loss)

        assert checkpoint.worst_of_best_epochs() == 0
        assert (
            len(cv_trainer_with_setup.current_ckptfiles())
            == 3 * cv_trainer_with_setup.num_folds
        )

        ## now add a better epoch
        ## this will kick out epoch 0 with score 0.5
        cv_trainer_with_setup.current_epoch = cv_trainer_with_setup.current_epoch + 1
        current_val_metrics = {"loss": torch.tensor(0.1)}
        cv_trainer_with_setup.current_val_metrics = current_val_metrics
        checkpoint._save_topk_ckpt(
            cv_trainer_with_setup, current_val_metrics["loss"].item()
        )

        curr_ckptfiles = cv_trainer_with_setup.current_ckptfiles()
        assert all("0.5" not in ckpt.name for ckpt in curr_ckptfiles)

        assert checkpoint.worst_of_best_epochs() == 1

        # now let's try to add the new worst epoch
        # this should not change the worst of best epochs
        cv_trainer_with_setup.current_epoch = cv_trainer_with_setup.current_epoch + 1
        current_val_metrics = {"loss": torch.tensor(1.2)}
        cv_trainer_with_setup.current_val_metrics = current_val_metrics
        checkpoint._save_topk_ckpt(
            cv_trainer_with_setup, current_val_metrics["loss"].item()
        )

        assert cv_trainer_with_setup.current_ckptfiles() == curr_ckptfiles
        assert checkpoint.worst_of_best_epochs() == 1
