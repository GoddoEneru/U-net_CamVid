import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
# from aim.pytorch_lightning import AimLogger

from lightning_wrapper import LightningWrapper


def train(cfg, camvid):
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    lt_wrapper = LightningWrapper(num_classes=cfg.params.num_classes,
                                  learning_rate=cfg.params.lr)
    lt_wrapper.to(device)   # A verifier

    # logger = AimLogger(experiment='segmentation_unet', log_system_params=False)
    trainer = pl.Trainer(max_epochs=cfg.params.epoch,
                         # logger=logger,
                         callbacks=[EarlyStopping(monitor=cfg.params.early_stop.metric,
                                                  mode=cfg.params.early_stop.mode,
                                                  min_delta=cfg.params.early_stop.delta,
                                                  patience=cfg.params.early_stop.patience,
                                                  verbose=cfg.params.early_stop.verbose)])
    trainer.fit(lt_wrapper, camvid)
    print("Training Done!")
    return lt_wrapper
