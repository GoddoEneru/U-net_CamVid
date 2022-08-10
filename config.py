from dataclasses import dataclass


@dataclass
class EarlyStop:
    mode: str
    patience: int
    delta: float
    metric: str
    verbose: bool


@dataclass
class Params:
    num_classes: int
    lr: float
    epoch: int
    batch_size: int
    early_stop: EarlyStop


@dataclass
class System:
    gpu: int
    workers: int
    parallel: str


@dataclass
class Dataset:
    train: str
    train_mask: str
    val: str
    val_mask: str
    test: str
    test_mask: str
    classes: str


@dataclass
class UnetConfig:
    params: Params
    system: System
    dataset: Dataset
