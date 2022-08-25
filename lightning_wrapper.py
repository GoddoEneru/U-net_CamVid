import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from model import NeuralNetwork


class LightningWrapper(pl.LightningModule):
    def __init__(self, num_classes, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.model = NeuralNetwork(num_classes)
        self.learning_rate = learning_rate

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        (x, y) = (batch['img'].to(self.device), batch['mask_machine'].to(self.device))
        x = F.normalize(x)
        pred = self(x)
        loss = F.cross_entropy(pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        (x, y) = (batch['img'].to(self.device), batch['mask_machine'].to(self.device))
        x = F.normalize(x)
        pred = self(x)
        val_loss = F.cross_entropy(pred, y)
        self.log("val_loss", val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        (x, y) = (batch['img'].to(self.device), batch['mask_machine'].to(self.device))
        x = F.normalize(x)
        pred = self(x)
        test_loss = F.cross_entropy(pred, y)
        self.log("test_loss", test_loss)
        return test_loss

    def forward(self, x):
        return self.model(x)
