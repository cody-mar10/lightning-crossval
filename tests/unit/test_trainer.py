from collections import defaultdict
from functools import partial
from pathlib import Path

import lightning
import pytest
import torch
from conftest import NUM_GROUPS, DummyModel, DummyModelConfig, _add_callback
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR

from lightning_cv.callbacks import Callback
from lightning_cv.config import CrossValidationTrainerConfig
from lightning_cv.data import CrossValidationDataModule
from lightning_cv.trainer import CrossValidationTrainer, TrainerState
from lightning_cv.utils.stopping import StopReasons


class TestTrainerInit:
    from lightning.fabric.loggers.csv_logs import CSVLogger
    from lightning.fabric.loggers.tensorboard import TensorBoardLogger

    from lightning_cv.callbacks.checkpoint import ModelCheckpoint
    from lightning_cv.callbacks.summary import ModelSummary

    @pytest.mark.parametrize("logger", [None, TensorBoardLogger])
    def test_logger(self, cv_trainer_config: CrossValidationTrainerConfig, logger):
        if logger is None:
            expected_logger_cls = self.CSVLogger
        else:
            expected_logger_cls = logger
            logger_instance = logger(root_dir=cv_trainer_config.checkpoint_dir)
            cv_trainer_config.loggers = logger_instance

        trainer = CrossValidationTrainer(
            model_type=DummyModel, config=cv_trainer_config
        )

        assert isinstance(trainer.fabric.logger, expected_logger_cls)

    @pytest.mark.parametrize("gradient_clip_algorithm", ["norm", "value"])
    def test_gradient_clipping(
        self, cv_trainer_config: CrossValidationTrainerConfig, gradient_clip_algorithm
    ):
        cv_trainer_config.gradient_clip_algorithm = gradient_clip_algorithm
        trainer = CrossValidationTrainer(
            model_type=DummyModel, config=cv_trainer_config
        )

        if gradient_clip_algorithm == "norm":
            assert trainer.gradient_clip_kwargs == {
                "max_norm": cv_trainer_config.gradient_clip_val
            }
        else:
            assert trainer.gradient_clip_kwargs == {
                "clip_val": cv_trainer_config.gradient_clip_val
            }

    @pytest.mark.parametrize("add", ["checkpoint", "summary", None])
    def test_add_default_callbacks(
        self, cv_trainer_config: CrossValidationTrainerConfig, add
    ):
        model_checkpoint = self.ModelCheckpoint()
        model_summary = self.ModelSummary()

        if add == "checkpoint":
            cv_trainer_config.callbacks = model_checkpoint
        elif add == "summary":
            cv_trainer_config.callbacks = model_summary

        trainer = CrossValidationTrainer(
            model_type=DummyModel, config=cv_trainer_config
        )

        for callback in trainer.callbacks:
            if isinstance(callback, self.ModelSummary):
                if add == "summary":
                    assert callback is model_summary
                else:
                    assert callback is not model_summary

            if isinstance(callback, self.ModelCheckpoint):
                if add == "checkpoint":
                    assert callback is model_checkpoint
                else:
                    assert callback is not model_checkpoint


class TestTrainerProperties:

    def test_callbacks(self, cv_trainer_with_cleanup: CrossValidationTrainer):
        assert isinstance(cv_trainer_with_cleanup.callbacks, list)

    def test_current_val_metrics(self, cv_trainer_with_cleanup: CrossValidationTrainer):
        field = "_current_val_metrics"
        assert not hasattr(cv_trainer_with_cleanup, field)

        with pytest.raises(RuntimeError):
            cv_trainer_with_cleanup.current_val_metrics

        current_val_metrics = {"loss": torch.tensor(0.5)}
        cv_trainer_with_cleanup.current_val_metrics = current_val_metrics

        # kind of silly, but want to show that no runtime error is raised
        assert cv_trainer_with_cleanup.current_val_metrics == current_val_metrics

    @pytest.mark.parametrize(
        "logger_state", ["disabled", "change_version", "remove_root_dir"]
    )
    def test_checkpoint_dir(
        self, cv_trainer_with_cleanup: CrossValidationTrainer, logger_state: str
    ):
        cv_trainer_with_cleanup.config

        root_ckpt_dir = cv_trainer_with_cleanup.config.checkpoint_dir
        if logger_state == "disabled":
            cv_trainer_with_cleanup.fabric._loggers = []
            assert cv_trainer_with_cleanup.checkpoint_dir == root_ckpt_dir.joinpath(
                "expt_0/version_0"
            )

        elif logger_state == "change_version":
            # CSVLogger
            logger = cv_trainer_with_cleanup.fabric.logger
            logger._version = 2  # type: ignore
            assert cv_trainer_with_cleanup.checkpoint_dir == root_ckpt_dir.joinpath(
                "lightning_logs/version_2"
            )
        else:
            logger = cv_trainer_with_cleanup.fabric.logger
            logger._version = 0  # type: ignore
            logger._root_dir = None  # type: ignore
            assert cv_trainer_with_cleanup.checkpoint_dir == Path(
                "checkpoints/lightning_logs/version_0"
            )

    @pytest.mark.parametrize(
        "status", [StopReasons.NULL, StopReasons.NO_IMPROVEMENT, "INVALID_STATUS"]
    )
    def test_status(self, cv_trainer_with_cleanup: CrossValidationTrainer, status):
        cv_trainer_with_cleanup.should_stop = True

        if status == "INVALID_STATUS":
            with pytest.raises(TypeError):
                cv_trainer_with_cleanup.status = status
        else:
            cv_trainer_with_cleanup.status = status
            assert cv_trainer_with_cleanup.status == status

            if status != StopReasons.NULL:
                assert (
                    cv_trainer_with_cleanup.state_when_stopped == TrainerState.TRAINING
                )

    def test_num_folds(
        self,
        cv_trainer_with_cleanup: CrossValidationTrainer,
        datamodule: CrossValidationDataModule,
        model_config,
    ):
        assert cv_trainer_with_cleanup.num_folds == 0

        cv_trainer_with_cleanup.setup(datamodule, model_config)

        assert cv_trainer_with_cleanup.num_folds == NUM_GROUPS

    def test_fold_manager(
        self,
        cv_trainer_with_cleanup: CrossValidationTrainer,
        datamodule: CrossValidationDataModule,
        model_config,
    ):
        with pytest.raises(AttributeError):
            cv_trainer_with_cleanup.fold_manager

        cv_trainer_with_cleanup.setup(datamodule, model_config)

        assert cv_trainer_with_cleanup.num_folds == NUM_GROUPS

    def test_state_when_stopped(self, cv_trainer_with_cleanup: CrossValidationTrainer):
        cv_trainer_with_cleanup.current_state = TrainerState.VALIDATING

        assert cv_trainer_with_cleanup.state_when_stopped == TrainerState.NOT_STOPPED

        # haven't stopped yet
        with pytest.raises(ValueError):
            cv_trainer_with_cleanup.state_when_stopped = TrainerState.VALIDATING

        # now stop the trainer
        cv_trainer_with_cleanup.should_stop = True
        cv_trainer_with_cleanup.status = StopReasons.NO_IMPROVEMENT

        assert cv_trainer_with_cleanup.state_when_stopped == TrainerState.VALIDATING

        with pytest.raises(TypeError):
            cv_trainer_with_cleanup.state_when_stopped = "INVALID_STATUS"

    @pytest.mark.parametrize(
        "state",
        [
            TrainerState.TRAINING,
            TrainerState.VALIDATING,
            TrainerState.NOT_STOPPED,
            "INVALID_STATE",
        ],
    )
    def test_current_state(
        self, cv_trainer_with_cleanup: CrossValidationTrainer, state
    ):
        assert cv_trainer_with_cleanup.current_state == TrainerState.TRAINING

        if state == "INVALID_STATE":
            with pytest.raises(TypeError):
                cv_trainer_with_cleanup.current_state = state
        elif state == TrainerState.NOT_STOPPED:
            with pytest.raises(ValueError):
                cv_trainer_with_cleanup.current_state = state
        else:
            cv_trainer_with_cleanup.current_state = state
            assert cv_trainer_with_cleanup.current_state == state


class TestTrainerTrainingMethods:
    class NotLightningDataModuleButHasCVMethod:
        def train_val_dataloaders(self):
            pass

    class NotLightningDataModuleAndMissingCVMethod:
        pass

    class LightningDataModuleMissingCVMethod(lightning.LightningDataModule):
        pass

    @pytest.mark.parametrize("setup_complete", [True, False])
    def test_setup(
        self,
        cv_trainer_with_cleanup: CrossValidationTrainer,
        setup_complete: bool,
        datamodule: CrossValidationDataModule,
        model_config,
    ):
        cv_trainer_with_cleanup.setup_complete = setup_complete
        cv_trainer_with_cleanup.setup(datamodule, model_config)

        if setup_complete:
            # no setup should have occurred, so fold manager should not exist
            with pytest.raises(AttributeError):
                cv_trainer_with_cleanup.fold_manager
        else:
            assert cv_trainer_with_cleanup.num_folds == NUM_GROUPS

    @pytest.mark.parametrize(
        "datamodule_cls",
        [
            NotLightningDataModuleAndMissingCVMethod,
            NotLightningDataModuleButHasCVMethod,
            LightningDataModuleMissingCVMethod,
        ],
    )
    def test_invalid_datamodule(
        self, cv_trainer_with_cleanup: CrossValidationTrainer, datamodule_cls
    ):
        expected_error_message = "LightningDataModule: {is_lightning_datamodule}\nProvided module has method train_val_dataloaders: {has_cv_method}"

        if datamodule_cls is self.NotLightningDataModuleAndMissingCVMethod:
            match = expected_error_message.format(
                is_lightning_datamodule=False, has_cv_method=False
            )
        elif datamodule_cls is self.NotLightningDataModuleButHasCVMethod:
            match = expected_error_message.format(
                is_lightning_datamodule=False, has_cv_method=True
            )
        else:
            match = expected_error_message.format(
                is_lightning_datamodule=True, has_cv_method=False
            )

        datamodule = datamodule_cls()

        with pytest.raises(ValueError, match=match):
            cv_trainer_with_cleanup.train_with_cross_validation(datamodule, {})  # type: ignore[arg-type]

    def test_stop_cv_loop(
        self,
        cv_trainer_with_cleanup: CrossValidationTrainer,
        datamodule: CrossValidationDataModule,
        model_config,
        caplog,
    ):
        assert (
            cv_trainer_with_cleanup.current_epoch
            < cv_trainer_with_cleanup.config.max_epochs
        )

        # this should cause training to stop after 1 epoch
        cv_trainer_with_cleanup.current_epoch = (
            cv_trainer_with_cleanup.config.max_epochs
        )
        cv_trainer_with_cleanup.train_with_cross_validation(datamodule, model_config)

        assert (
            cv_trainer_with_cleanup.current_epoch
            == cv_trainer_with_cleanup.config.max_epochs + 1
        )

        assert "COMPLETE: Training 2 epochs completed" in caplog.text

    @pytest.mark.parametrize("grad_clip_algorithm", ["norm", "value"])
    @pytest.mark.parametrize("grad_clip_val", [0.0, 0.5])
    def test_clip_gradients(
        self,
        cv_trainer_config: CrossValidationTrainerConfig,
        datamodule: CrossValidationDataModule,
        model_config,
        grad_clip_algorithm,
        grad_clip_val: float,
    ):
        cv_trainer_config.gradient_clip_algorithm = grad_clip_algorithm
        cv_trainer_config.gradient_clip_val = grad_clip_val

        cv_trainer = CrossValidationTrainer(
            model_type=DummyModel, config=cv_trainer_config
        )

        if grad_clip_val == 0.0:
            assert not cv_trainer.apply_gradient_clipping
        else:
            cv_trainer.setup(datamodule, model_config)

            train_loader, _ = next(datamodule.train_val_dataloaders())
            fold = 0
            model = cv_trainer.fold_manager[fold].model
            optimizer = cv_trainer.fold_manager[fold].optimizer

            CURRENT_GRAD_VALUE = 2.0
            for param in model.parameters():
                param.grad = torch.full_like(
                    param, CURRENT_GRAD_VALUE, requires_grad=False
                )

            cv_trainer.fabric.clip_gradients(
                module=model,
                optimizer=optimizer,
                error_if_nonfinite=False,
                **cv_trainer.gradient_clip_kwargs,
            )

            if grad_clip_algorithm == "value":
                for param in model.parameters():
                    assert torch.all(param.grad == grad_clip_val)
            else:
                for param in model.parameters():
                    assert torch.norm(param.grad) <= grad_clip_val

    class OptimizerStepCounter(Callback):
        def __init__(self):
            self.num_optimizer_steps_per_fold = defaultdict(int)

        def on_before_optimizer_step(
            self,
            trainer: CrossValidationTrainer,
            optimizer: Optimizer,
            optimizer_idx: int = 0,
        ):
            current_fold = trainer.current_fold
            self.num_optimizer_steps_per_fold[current_fold] += 1

    @pytest.mark.parametrize("grad_accum_steps", [1, 4, 1000])
    def test_gradient_accumulation(
        self,
        cv_trainer_with_cleanup: CrossValidationTrainer,
        datamodule: CrossValidationDataModule,
        model_config,
        grad_accum_steps: int,
    ):
        callback = self.OptimizerStepCounter()
        _add_callback(cv_trainer_with_cleanup, callback)
        cv_trainer_with_cleanup.config.grad_accum_steps = grad_accum_steps

        cv_trainer_with_cleanup.train_with_cross_validation(datamodule, model_config)

        # there are 30 data points, with 10 belonging to each class, meaning that there should be 3 folds
        # the batch size is 5, so each fold should have 20 total data points, divided into 4 batches
        # and there are 2 epochs, so the total number of optimizer steps should
        # thus, the expected number of optimizer steps is:
        # grad_accum_steps = 1: 8
        # grad_accum_steps = 4: 2
        # grad_accum_steps = 1000: 0

        EXPECTED_NUM_OPTIMIZER_STEPS = {1: 8, 4: 2, 1000: 0}
        expected_num_optimizer_steps = EXPECTED_NUM_OPTIMIZER_STEPS[grad_accum_steps]

        assert all(
            callback.num_optimizer_steps_per_fold[fold] == expected_num_optimizer_steps
            for fold in range(cv_trainer_with_cleanup.num_folds)
        )

    class InvalidModel(lightning.LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(2, 1)

        def training_step(self, batch, batch_idx):
            # computes loss and then does something with it like logging
            # but does not return loss from this method
            pass

        def validation_step(self, batch, batch_idx):
            # computes loss and then does something with it like logging
            # but does not return loss from this method
            pass

    def test_training_step_invalid_model(
        self, cv_trainer_with_cleanup: CrossValidationTrainer
    ):
        with pytest.raises(RuntimeError):
            cv_trainer_with_cleanup.training_step(
                model=self.InvalidModel(), batch=(), batch_idx=0
            )

    def test_validation_step_invalid_model(
        self, cv_trainer_with_cleanup: CrossValidationTrainer
    ):
        model = self.InvalidModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # need to set current epoch on the TqdmProgressBar
        from lightning_cv.callbacks.progress import TqdmProgressBar

        for callback in cv_trainer_with_cleanup.fabric._callbacks:
            if isinstance(callback, TqdmProgressBar):
                callback.current_epoch = 0

        with pytest.raises(RuntimeError):
            cv_trainer_with_cleanup.val_loop(
                model=model,
                val_loader=list(range(5)),  # type: ignore[arg-type] <- expecting failure
                optimizer=optimizer,
            )

    @pytest.mark.parametrize(
        ("current_epoch", "validation_frequency", "expected"),
        [
            (0, 1, True),
            (1, 1, True),
            (0, 2, False),
            (1, 2, True),
        ],
    )
    def test_should_validate(
        self,
        cv_trainer_with_cleanup: CrossValidationTrainer,
        current_epoch: int,
        validation_frequency: int,
        expected: bool,
    ):
        cv_trainer_with_cleanup.current_epoch = current_epoch
        cv_trainer_with_cleanup.config.validation_frequency = validation_frequency

        assert cv_trainer_with_cleanup.should_validate is expected

    @staticmethod
    def _test_configure_optimizers_values():
        model = DummyModel(DummyModelConfig(10))
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

        test_values = [
            (optimizer, (optimizer, None)),
            ({"optimizer": optimizer}, (optimizer, None)),
            (
                {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}},
                (
                    optimizer,
                    {
                        "scheduler": scheduler,
                        "interval": "epoch",
                        "frequency": 1,
                        "monitor": "val_loss",
                    },
                ),
            ),
            ("INVALID", "ERROR"),
            ((optimizer, scheduler), (optimizer, scheduler)),
            ([optimizer, optimizer], "ERROR"),
            (
                {
                    "optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "frequency": 222},
                },
                (
                    optimizer,
                    {
                        "scheduler": scheduler,
                        "interval": "epoch",
                        "frequency": 222,
                        "monitor": "val_loss",
                    },
                ),
            ),
            (
                (optimizer, {"scheduler": scheduler}),
                (
                    optimizer,
                    {
                        "scheduler": scheduler,
                        "interval": "epoch",
                        "frequency": 1,
                        "monitor": "val_loss",
                    },
                ),
            ),
            (
                (optimizer, {"scheduler": scheduler, "frequency": 123}),
                (
                    optimizer,
                    {
                        "scheduler": scheduler,
                        "interval": "epoch",
                        "frequency": 123,
                        "monitor": "val_loss",
                    },
                ),
            ),
        ]

        return pytest.mark.parametrize(
            ("configure_optim_output", "expected"), test_values
        )

    @_test_configure_optimizers_values()
    def test_parse_optimizers_schedulers(
        self,
        cv_trainer_with_cleanup: CrossValidationTrainer,
        configure_optim_output,
        expected,
    ):
        lazy_eval = partial(
            cv_trainer_with_cleanup._parse_optimizers_schedulers, configure_optim_output
        )

        evaluate = False
        error = BaseException

        if isinstance(configure_optim_output, dict):
            if "optimizer" not in configure_optim_output:
                error = KeyError
            else:
                evaluate = True
        elif isinstance(configure_optim_output, Optimizer) or isinstance(
            configure_optim_output, tuple
        ):
            evaluate = True
        else:
            error = ValueError

        if evaluate:
            output = lazy_eval()
            assert output == expected
        else:
            with pytest.raises(error):
                lazy_eval()

    def test_invalid_callback(self, cv_trainer_with_cleanup: CrossValidationTrainer):
        with pytest.raises(RuntimeError):
            cv_trainer_with_cleanup.apply_callback("INVALID_CALLBACK")

    @staticmethod
    def _test_step_scheduler():
        gamma = 0.1
        init_lr = DummyModel.LR

        scheduler_every_step = {"step_size": 1, "gamma": gamma}

        scheduler_every_2_steps = {"step_size": 2, "gamma": gamma}

        test_values = [
            (None, [init_lr]),
            (
                {
                    "builder": scheduler_every_step,
                    "frequency": 1,
                    "interval": "epoch",
                    "monitor": "train_loss",
                },
                [init_lr, init_lr * gamma],
            ),
            (
                {
                    "builder": scheduler_every_step,
                    "frequency": 2,
                    "interval": "epoch",
                    "monitor": "train_loss",
                },
                [init_lr, init_lr, init_lr * gamma],
            ),
            (
                {
                    "builder": scheduler_every_2_steps,
                    "frequency": 1,
                    "interval": "epoch",
                    "monitor": "train_loss",
                },
                [init_lr, init_lr, init_lr * gamma],
            ),
            (
                {
                    "builder": scheduler_every_2_steps,
                    "frequency": 2,
                    "interval": "epoch",
                    "monitor": "train_loss",
                },
                # epoch 0,      1,       2,       3,       4,
                # freq NA,      0,    STEP,       0,    STEP
                # sch NA,       0,       1,       1,    STEP
                [init_lr, init_lr, init_lr, init_lr, init_lr * gamma],
            ),
        ]

        return pytest.mark.parametrize(("scheduler", "expected_lr"), test_values)

    class _ReportLR(Callback):
        def __init__(self):
            self.lrs = []

        def on_train_fold_end(self, trainer: CrossValidationTrainer):
            fold_idx = 0
            fold_state = trainer.fold_manager[fold_idx]
            optimizer = fold_state.optimizer
            scheduler = fold_state.scheduler

            if scheduler is not None:
                lr = scheduler["scheduler"].get_lr()[0]  # type: ignore[attr-defined]
            else:
                lr = optimizer.param_groups[0]["lr"]

            self.lrs.append(float(lr))

    # TODO: this isn't working for some reason....
    @pytest.mark.skip()
    @_test_step_scheduler()
    @pytest.mark.parametrize("interval", ["step", "epoch"])
    def test_step_scheduler(
        self,
        preset_cv_trainer_with_cleanup: CrossValidationTrainer,
        scheduler,
        expected_lr: list[float],
        interval: str,
        datamodule,
    ):
        if interval == "step":
            expected_lr = [DummyModel.LR] * len(expected_lr)

        epochs = len(expected_lr) - 1
        epochs = max(epochs, 2)
        preset_cv_trainer_with_cleanup.config.max_epochs = epochs

        fold = 0
        model = preset_cv_trainer_with_cleanup.fold_manager[fold].model
        optimizer = preset_cv_trainer_with_cleanup.fold_manager[fold].optimizer

        # preset_cv_trainer_with_cleanup.current_val_metrics = {
        #     "val_loss": torch.tensor(0.5)
        # }

        # assert self._get_current_lr(None) == DummyModel.LR

        if scheduler is not None:
            scheduler["scheduler"] = StepLR(optimizer, **scheduler["builder"])
            preset_cv_trainer_with_cleanup.fold_manager[fold].scheduler = scheduler

        lr_monitor = self._ReportLR()
        _add_callback(preset_cv_trainer_with_cleanup, lr_monitor)

        # lazy_step_scheduler = partial(
        #     preset_cv_trainer_with_cleanup.step_scheduler,
        #     model=model,  # type: ignore[arg-type]
        #     scheduler_cfg=scheduler,
        #     level=interval,  # type: ignore[arg-type]
        # )

        # if interval == "epoch":  # same as test cases
        #     # obs_lr = []
        # for lr in range(len(expected_lr) + 1):
        #     for batch_idx, batch in enumerate(dataloader):
        #         loss = model.training_step(batch, 0)
        #     optimizer.step()

        #     # on epoch 0, this should just set the lr to the initial value
        #     lazy_step_scheduler(
        #         current_value=preset_cv_trainer_with_cleanup.current_epoch
        #     )
        #     # assert self._get_current_lr(optimizer) == lr
        #     if scheduler is not None:
        #         lr = self._get_current_lr(scheduler["scheduler"])
        #     else:
        #         lr = self._get_current_lr(None)

        #     obs_lr.append(lr)
        #     preset_cv_trainer_with_cleanup.current_epoch += 1

        # obs_lr = torch.tensor(obs_lr)
        # expected_lr.insert(0, DummyModel.LR)
        # expected_lr = torch.tensor(expected_lr)
        # assert torch.allclose(obs_lr, expected_lr)
        preset_cv_trainer_with_cleanup.train_with_cross_validation(
            datamodule, None
        )  # already setup so don't need model config

        assert lr_monitor.lrs == expected_lr


class TestTrainerUtils:
    from lightning_cv.config import FoldState

    @pytest.mark.parametrize(
        ("limit_batches", "expected"), [(0.25, 5), (0.5, 10), (1.0, 21), (10, 10)]
    )
    def test_limit_batches(self, limit_batches, expected: int):
        total_batches = 21
        limit = CrossValidationTrainer._limit_batches(total_batches, limit_batches)

        assert limit == expected

    @pytest.mark.parametrize(
        ("status", "expected_message"),
        [
            (StopReasons.NULL, "COMPLETE: Training 2 epochs completed."),
            (
                StopReasons.NO_IMPROVEMENT,
                "EARLY STOP: Training stopped early due to no improvement in val_loss",
            ),
            (
                StopReasons.TIMEOUT,
                "TIMEOUT: Training stopped early due to running out of time.",
            ),
            (
                StopReasons.PRUNED_TRIAL,
                "PRUNED: Training stopped early due to bad trial pruning during hyperparameter "
                "tuning.",
            ),
            (
                StopReasons.LOSS_NAN_OR_INF,
                "NAN/INF: Training stopped early due to NaN or Inf loss.",
            ),
            (
                StopReasons.PERFORMANCE_STALLED,
                "STALLED: Training stopped early due to performance stalling. val_loss is not "
                "improving during the monitored stage.",
            ),
        ],
    )
    @pytest.mark.parametrize(
        "trainer",
        ["rank_zero_cv_trainer_with_cleanup", "non_rank_zero_cv_trainer_with_cleanup"],
    )
    def test_log_message(
        self,
        trainer: str,
        status: StopReasons,
        expected_message: str,
        request,
        caplog,
    ):
        cv_trainer_with_cleanup: CrossValidationTrainer = request.getfixturevalue(
            trainer
        )
        cv_trainer_with_cleanup.should_stop = True
        cv_trainer_with_cleanup.status = status
        cv_trainer_with_cleanup.log_status()

        if trainer == "rank_zero_cv_trainer_with_cleanup":
            assert cv_trainer_with_cleanup.is_global_zero
        else:
            assert not cv_trainer_with_cleanup.is_global_zero

        if cv_trainer_with_cleanup.is_global_zero:
            assert expected_message in caplog.text
        else:
            assert expected_message not in caplog.text

    def test_load_checkpoint(
        self,
        preset_cv_trainer_with_cleanup: CrossValidationTrainer,
        checkpoint_file: Path,
    ):
        loaded_ckpt = preset_cv_trainer_with_cleanup.load(checkpoint_file)

        assert isinstance(loaded_ckpt, self.FoldState)
        assert torch.all(loaded_ckpt.model.predictor.weight == 2.0)

        # these were preset when creating this test
        assert preset_cv_trainer_with_cleanup.current_epoch == 5
        assert preset_cv_trainer_with_cleanup.current_fold == 2
        assert preset_cv_trainer_with_cleanup.global_step_per_fold[2] == 28
