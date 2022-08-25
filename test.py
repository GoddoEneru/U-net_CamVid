# import torch
import pytorch_lightning as pl
# from utils import show_img


def test(cfg, lt_wrapper, trainer, camvid):
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer.test(lt_wrapper, camvid)
    print("Testing Done!")

    # data = test_data[0]
    # img = data['img'].to(device).unsqueeze(0)
    # pred = lt_wrapper.model.to(device).predict(img)
    # show_img(pred, data)
