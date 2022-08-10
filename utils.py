import torch
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt


def show_img(pred, data):
    f, axarr = plt.subplots(1, 3)
    pred = torch.argmax(pred, dim=1)
    axarr[0].imshow(to_pil_image(data['img']))
    axarr[1].imshow(to_pil_image(data['mask_machine'].int()))
    axarr[2].imshow(to_pil_image(pred.float()))
    plt.show()
