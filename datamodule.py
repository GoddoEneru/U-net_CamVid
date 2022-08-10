import pytorch_lightning as pl
import pandas as pd
from typing import Optional
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from custom_dataset import CustomDataset


class Datamodule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.classes = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def prepare_data(self):
        self.classes = pd.read_csv(self.cfg.dataset.classes)

    def setup(self, stage: Optional[str] = None):
        self.train_data = CustomDataset(self.cfg.dataset.train, self.cfg.dataset.train_mask, self.classes)
        self.val_data = CustomDataset(self.cfg.dataset.val, self.cfg.dataset.val_mask, self.classes)
        self.test_data = CustomDataset(self.cfg.dataset.test, self.cfg.dataset.test_mask, self.classes)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_dataloader = DataLoader(self.train_data, batch_size=self.cfg.params.batch_size)
        return train_dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        val_dataloader = DataLoader(self.val_data, batch_size=self.cfg.params.batch_size)
        return val_dataloader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        test_dataloader = DataLoader(self.test_data, batch_size=self.cfg.params.batch_size)
        return test_dataloader
