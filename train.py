import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
# from aim.pytorch_lightning import AimLogger


def train(cfg, camvid, lt_wrapper, logger):
    # logger = AimLogger(experiment='segmentation_unet', log_system_params=False)
    trainer = pl.Trainer(max_epochs=cfg.params.epoch,
                         logger=logger,
                         accelerator='gpu',
                         devices=1,
                         callbacks=[EarlyStopping(monitor=cfg.params.early_stop.metric,
                                                  mode=cfg.params.early_stop.mode,
                                                  min_delta=cfg.params.early_stop.delta,
                                                  patience=cfg.params.early_stop.patience,
                                                  verbose=cfg.params.early_stop.verbose)])
    trainer.fit(lt_wrapper, camvid)
    print("Training Done!")
    return lt_wrapper, trainer
