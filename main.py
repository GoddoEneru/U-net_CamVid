import hydra
from hydra.core.config_store import ConfigStore
from pytorch_lightning.loggers import WandbLogger
import wandb
from dotenv import load_dotenv
import os

from config import UnetConfig
from datamodule import Datamodule
from train import train
from test import test
from lightning_wrapper import LightningWrapper


load_dotenv()
cs = ConfigStore.instance()
cs.store(name='unet_config', node=UnetConfig)


@hydra.main(config_path='conf', config_name='config', version_base=None)
def app(cfg: UnetConfig):
    camvid = Datamodule(cfg)

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using {device} device")
    lt_wrapper = LightningWrapper(num_classes=cfg.params.num_classes,
                                  learning_rate=cfg.params.lr)
    # lt_wrapper.to(device)

    wandb.login(key=os.getenv("WANDB_KEY"))
    logger = WandbLogger(project="test_wandb")

    lt_wrapper, trainer = train(cfg, camvid, lt_wrapper, logger)
    test(cfg, lt_wrapper, trainer, camvid)


if __name__ == "__main__":
    app()
