**Cross validation with [Lightning Fabric](https://lightning.ai/docs/fabric/stable/)**
===========

# Installation
```bash
pip install lightning-cv
```

# Purpose
[PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) provides many boilerplate-free abstractions to building scalable deep learning models. However, there is little first-class support for cross validation, which is very common when training models with many hyperparameters and heterogeneous data.

Fortunately, the creators of [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) have developed and released a more flexible package that underlies `PyTorch Lightning` called [Lightning Fabric](https://lightning.ai/docs/fabric/stable/). `Fabric` automatically handles lower technical details like target training device, training precision, and distributed training. BUT `Fabric` leaves the flexibility of setting up training/inference loops, which is beneficial for developing a general purpose cross validation scheme.

# Overall workflow
The cross validation logic can be described with this pseudocode:

```python
import torch

n_folds = len(folds)

models = [
    (MyModel(), MyOptimizer()) for _ in range(n_folds)
]

# tensor that is n_folds large and keeps track of per-fold loss
per_fold_loss = torch.zeros((n_folds,))
for epoch in range(epochs):
    for fold, (fold_train_loader, fold_val_loader) in enumerate(folds):
        model, optimizer = models[fold]

        # train loop -> step level
        for batch in fold_train_loader:
            # forward pass
            loss = model(batch)

            # backward pass and optimizer update
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # val loop -> step level
        for batch in fold_val_loader:
            loss = model(batch)
            # keep running total at the fold level
            per_fold_loss[fold] += loss
        
        # take avg of val loss for current fold
        per_fold_loss[fold] /= len(fold_val_loader)
    
    # at this point, all folds have been trained AND validated once
    # therefore, we can compute a current val loss as the avg of the
    # per-fold loss
    current_val_loss = per_fold_loss.mean()

    # this is useful since we can monitor cross val performance in real-time
    # here, the report function can do several things:
    #   1. log performance
    #   2. model checkpointing
    #   3. early stopping
    #   4. make use of a scheduler
    report(current_val_loss)

    # reset per_fold_loss
    per_fold_loss.fill_(0.0) 
```

<details>
  <summary>Details</summary>

## Fold Synchronization
Notably, this cross validation scheme **synchronizes all model folds at the epoch-level**. Unfortunately, this requires *k*-fold models to be held in memory simultaneously, which might be prohibitive for extremely large models. While looping over the training folds in the outermost loop could be optionally added and reduce the memory footprint, the current cross validation scheme offers greater flexibility.

The simplest benefit is that an individual cross validation run will be faster by a factor of ~*k* since each model fold is being trained within the epoch-level loop. However, a greater benefit of this approach is that this allows for early stopping of unpromising trials since the current validation metrics can be updated in real-time at the end of each epoch.

### Epoch level vs step level monitoring
Syncing folds at the epoch level means that techniques that *meaningfully* monitor current model performance **must** only operate at the **epoch** level. For example, schedulers can adjust the learning rate based on predefined strategies. The scheduler adjustment frequency, in normal workflows, can happen at the epoch or batch-step level. However, for this cross validation scheme, to ensure more comparable analysis across model folds, schedulers are only allowed to operate at the epoch level, so the learning rate is the same across all model folds.

A more explicit way to state this is that each training fold is not necessarily the same size, meaning that each training fold **could** take a different number of training steps. This becomes problematic when trying to monitor performance at the step level where the model folds are not synced.

The exception to this rule is when current performance is read without causing any meaningful updates. This occurs during metric logging and when a progress bar is used.

</details>

## Usage
<details>
  <summary>Data</summary>

The biggest change that needs to be made to existing workflows is to use a `lightning.LightningDataModule` subclass that has the method `train_val_dataloaders`. This method yields the train/val splits of the original dataset.

Convenience classes have been provided by this library:

```python
from typing import Any, Callable, Optional

import lightning as L
from torch.utils.data import Dataset

from lightning_cv._typing import CrossValidatorType

class _CrossValidationDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        
        # pass a class type that implements the sklearn split API
        # ie calling the instance's .split() method yields train/val indices
        cross_validator: type[CrossValidatorType],
        # any init kwargs required to instantiate the cross_validator
        cross_validator_kwargs: dict[str, Any],

        # optional torch.utils.data.DataLoader collate_fn
        collate_fn: Optional[Callable] = None,
    ):
        super().__init__()
        # store other input args as attrs
        self.dataset = dataset
        self.batch_size = batch_size
        self._cross_validator = cross_validator
        self._cross_validator_kwargs = cross_validator_kwargs
        self.collate_fn = collate_fn
    
    # this method can be overridden for a general-use datamodule
    def setup(self, stage):
        # cross validation only needs the fit stage
        if stage == "fit":
            self.data_manager = self._cross_validator(**self._cross_validator_kwargs)

    def train_val_dataloaders(self):
        for train_idx, val_idx in self.data_manager.split():
            # convert train/val data indices into a DataLoader
            # actual impl depends on original dataset
            yield train_loader, val_loader
```
</details>

<details>
  <summary>Model</summary>

The model requirements are: 

| Requirement            | Method | Provided by `lightning.LightningModule` |
|------------------------|--------|-----------------------------------------|
| `training_step`        | Y      | Y                                       |
| `validation_step`      | Y      | Y                                       |
| `configure_optimizers` | Y      | Y                                       |
| `lr_scheduler_step`    | Y      | Y                                       |
| `train`                | Y      | Y                                       |
| `eval`                 | Y      | Y                                       |
| `fabric`               | N      | Y (technically, still need to set attr) |
| `estimated_steps`      | N      | N                                       |

## Extend existing models
Most of these method or attribute requirements are provided by creating models that are subclasses of the `lightning.LightningModule` class. A mixin class is provided that can handle the `fabric` and `estimated_steps` attributes to extend an existing `LightningModule`.

The `CrossValModuleMixin` provides the `estimated_steps` property and ensures that an instance of `lightning.Fabric` is passed.

TODO: mention pydantic model config

```diff
import lightning as L
+ from lightning_cv import CrossValModuleMixin

- class MyModel(L.LightningModule)
+ class MyModel(L.LightningModule, CrossValModuleMixin):
    def __init__(self, ...):
-       super().__init__(self)
+       L.LightningModule.__init__(self, ...)
+       CrossValModuleMixin.__init__(self, ...)
```

This way your model can be used with either the fully abstracted `lightning.Trainer` or the `lightning_cv.CrossValidationTrainer`.

## Create new models
Alternatively, if you are developing a new model from scratch, you can just subclass `CrossValModule`, which already subclasses `lightning.LightningModule` and the `CrossValModuleMixin`:

```python
from lightning_cv import CrossValModule:

class MyModel(CrossValModule):
    def __init__(self, ...):
        # handles L.LightningModule init and 
        # provides other api requirements
        super().__init__(...) 
        # custom init logic here
```
</details>

<details>
  <summary>Training</summary>

### Config
The `CrossValidationTrainer` can *only* be instantiated with a `pydantic` config model. `pydantic` models provide automatic type validation and safety guarantees that are useful for general purpose computing.

The config model looks like this. Most of these arguments get passed directly to `lightning.Fabric`, but several are general model training parameters that need to be tracked.

```python
from pydantic import BaseModel

class CrossValidationTrainerConfig(BaseModel):
    accelerator: Accelerators | Accelerator = "auto"
    strategy: Strategies | Strategy = "auto"
    devices: list[int] | str | int = "auto"
    precision: Precision | int = "32"
    plugins: Optional[str | Any] = None
    callbacks: Optional[list[Callback] | Callback] = None
    loggers: Optional[Logger | list[Logger]] = None
    max_epochs: int = 1000
    grad_accum_steps: int = 1
    limit_train_batches: Number = 1.0
    limit_val_batches: Number = 1.0
    validation_frequency: int = 1
    use_distributed_sampler: bool = True
    checkpoint_dir: Path = Path.cwd().joinpath("checkpoints")
    checkpoint_frequency: int = 1
    monitor: str = "val_loss"
```

You can instantiate the config model like this:

```python
from lightning_cv import CrossValidationTrainerConfig

# use defaults
trainer_config = CrossValidationTrainerConfig()

# change defaults
trainer_config = CrossValidationTrainerConfig(max_epochs=2)
```

### Trainer class
The trainer class (`CrossValidationTrainer`) only accepts two init arguments:
a model *type*, and the config instance.

Furthemore, the main method on the trainer is the `train_with_cross_validation`, which takes as input, a cross validation data module that has the `train_val_dataloaders` method, and a model config.

NOTE: Currently, the model config is also a `pydantic` model, but this requirement could be relaxed in the future.

```python
import lightning as L
from lightning_cv import CrossValidationTrainerConfig, CrossValModule

class CrossValidationTrainer:
    __fabric_keys__ = {
        "accelerator",
        "strategy",
        "devices",
        "precision",
        "plugins",
        "callbacks",
        "loggers",
    }

    def __init__(
        self, 
        model_type: type[CrossValModule], 
        config: CrossValidationTrainerConfig,
    ):  
        self.model_type
        self.config = config
        self.fabric = L.Fabric(**self.config.dict(include=self.__fabric_keys__))

        # other init logic

    def train_with_cross_validation(self, datamodule, model_config):
        # this method setups up k-fold models
        # then does the cross validation loop described in the 
        # `Overall workflow` section above
```

### Callbacks
`PyTorch Lightning` has predefined hooks that enable custom logic to be applied at various steps of the train/inference loop. In `Lightning Fabric`, no predefined hooks are automatically available, which provides authors flexibility of of defining what hooks are available.

The following hooks are available when using the `CrossValidationTrainer`:

| Hook                               | Called                                                               |
|------------------------------------|----------------------------------------------------------------------|
| on_train_start                     | **only once** when at the beginning of `train_with_cross_validation` |
| on_train_end                       | **only once** when at the end of `train_with_cross_validation`       |
| on_train_fold_start                | **per epoch** before the fold level loops                            |
| on_train_fold_end                  | **per epoch** after the fold level loops                             |
| on_train_epoch_start_per_fold      | **per epoch per fold** before batch level loops                      |
| on_train_epoch_end_per_fold        | **per epoch per fold** after batch level loops                       |
| on_train_batch_start_per_fold      | **per batch** before forward/backward, logging, etc                  |
| on_train_batch_end_per_fold        | **per batch** after forward/backward, logging, etc                   |
| on_before_optimizer_step           | **per batch** before `optimizer.step`                                |
| on_before_zero_grad                | **per batch** before `optimizer.zero_grad`                           |
| on_validation_start_per_fold       | **per epoch per fold** before batch level loops                      |
| on_validation_end_per_fold         | **per epoch per fold** after batch level loops                       |
| on_before_log_metrics              | **per batch** before metrics are logged during BOTH train/val loops  |
| on_validation_batch_start_per_fold | **per batch** before validation metrics are obtained                 |
| on_validation_batch_end_per_fold   | **per batch** after validation metrics are obtained                  |
| on_before_backward                 | **per batch** before `loss.backward`                                 |
| on_after_backward                  | **per batch** after `loss.backward`                                  |

A callback base class has been provided to to enable the hooks for any method. The `lightning_cv.callbacks.Callback` MUST be used to provide the hooks, since the `pydantic` trainer config will only accept subclasses of this exact class. 

An example callback to summarize model parameters that relies on a `lightning` function:

```python
import sys
from typing import cast

import lightning_cv as lcv
from lightning.pytorch.utilities.model_summary.model_summary import summarize


class ModelSummary(lcv.callbacks.Callback):
    def __init__(self, max_depth: int = 1):
        self.max_depth = max_depth

    # we only need to generate the model summary once at the beginning of cross validation
    def on_train_start(self, trainer: lcv.CrossValidationTrainer):
        model = trainer.fold_manager[0].model
        summary = summarize(model, max_depth=self.max_depth)
        sys.stderr.write(repr(summary))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_depth={self.max_depth})"
```

Then, any custom callbacks can be passed to the `callbacks` argument of the `CrossValidationTrainerConfig`.

</details>

<details>
  <summary>Minimal cross validation reference</summary>

```python
from lightning_cv import (
    CrossValidationTrainer, 
    CrossValidationTrainerConfig,
    CrossValModule,
    _CrossValDataModule
)
from pydantic import BaseModel

class MyModelConfig:
    ...

class MyModel(CrossValModule):
    def __init__(self, config: MyModelConfig, ...):
        super().__init__(...)
        ...
    
    def training_step(self, ...): ...
    def validation_step(self, ...): ...
    def configure_optimizers(self, ...): ...

class MyDataModule(_CrossValDataModule):
    def __init__(self, ...): ...
    def train_val_dataloaders(self, ...): ...

# 1. setup datamodule
datamodule = MyDataModule(...)

# 2. setup model init config
model_config = MyModelConfig()

# 3. setup trainer
config = CrossValidationTrainerConfig()
trainer = CrossValidationTrainer(model_type=MyModel, config=config)

# 4. run cross validation
#    handles setup logic for model, datamodule, and cv-folds
trainer.train_with_cross_validation(
    datamodule=datamodule,
    model_config=model_config
)
```
</details>

<details>
  <summary>Tuning</summary>

Currently, there is a simple integration for [optuna](https://optuna.readthedocs.io/en/stable/), a machine learning hyperparameter tuning library.

Here is a simple example that uses the provided `Tuner` class to setup tuning runs. Suppose you want to tune the learning rate:

```python
import optuna

from lightning_cv import (
    CrossValidationTrainer, 
    CrossValidationTrainerConfig,
    CrossValModule,
    _CrossValDataModule
)
from lightning_cv.tuning import Tuner
from pydantic import BaseModel
from functools import partial

class MyModelConfig:
    lr: float
    ...

class MyModel(CrossValModule):
    def __init__(self, config: MyModelConfig, ...):
        super().__init__(...)
        self.config = config
        ...
    
    def training_step(self, ...): ...
    def validation_step(self, ...): ...
    def configure_optimizers(self, ...):
        lr = self.config.lr

class MyDataModule(_CrossValDataModule):
    def __init__(self, ...): ...
    def train_val_dataloaders(self, ...): ...

# 1. setup datamodule
datamodule = MyDataModule(...)

# 2. setup model init config
model_config = MyModelConfig()

# 3. create a tuner
tuner = Tuner(
    model_type=MyModel,
    model_config=MyModelConfig(),
    datamodule=datamodule,
    trainer_config=CrossValidationTrainerConfig(),
)

# 4. create a tuning suggestion fn
#    signature: (optuna.Trial) -> dict[str, Any]
def get_trial_suggestions(trial: optuna.Trial):
    # the optuna trial object will sample a float
    # from the loguniform range [1e-4, 1e-2]
    lr = trial.suggest_float(name="lr", low=1e-4, high=1e-2, log=True)
    return {"lr": lr}

# 5. run tuning trials
study = optuna.create_study(
    study_name="test",
    # look into other args to modify study such as
    # bad trial pruning
)

# we still don't have any actual trial obj so need a partial fn
# that will take these later, thus the trial MUST 
# be the first arg
def _tuning_trial(trial: optuna.Trial, tuner: Tuner):
    return tuner.tune(trial=trial, func=get_trial_suggestions)

tuning_trial = partial(_tuning_trial, tuner=tuner)

study.optimize(
    tuning_trial,
    n_trials=100,
    direction="minimize" # ie min loss
)
```

The only public method the `Tuner` class has is the `tune` method. This method first adjusts the model config which new trialed values, which is only the learning rate in this example. All other default values are held constant. 

Then, the tuner sets up a trainer instance and calls the `train_with_cross_validation_method` with the datamodule. If the method finishes and is not pruned due to an unpromising trial, then the `.tune` method returns a monitored validation metric, such as the validation loss, averaged over all folds.

This validation metric is reported to the `optuna.Study` object, which then keeps track of the best run.

</details>