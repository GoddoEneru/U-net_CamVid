import hydra
from hydra.core.config_store import ConfigStore
from config import UnetConfig

from datamodule import Datamodule
from train import train
from test import test


cs = ConfigStore.instance()
cs.store(name='unet_config', node=UnetConfig)


@hydra.main(config_path='conf', config_name='config', version_base=None)
def app(cfg: UnetConfig):
    camvid = Datamodule(cfg)
    lt_wrapper = train(cfg, camvid)
    test(cfg, lt_wrapper, camvid)


if __name__ == "__main__":
    app()
