import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, img_path, mask_path, class_dict):
        self.transform = T.Compose([T.Resize(224), T.ToTensor()])
        self.img_path = img_path
        self.file_img = os.listdir(self.img_path)
        self.mask_path = mask_path
        self.file_mask = os.listdir(self.mask_path)
        self.class_dict = class_dict
        self.num_classes = len(class_dict)

    def __len__(self):
        return len(self.file_img)

    def __getitem__(self, idx):
        img = self.file_img[idx]
        img = Image.open(self.img_path + img)
        img = self.transform(img)
        mask = self.file_mask[idx]
        mask_img = Image.open(self.mask_path + mask)
        mask_img = T.Resize(224)(mask_img)

        mask = np.array(mask_img)
        mask_machine = []
        for i in range(self.num_classes):
            rgb_class = [self.class_dict.iloc[i]['r'], self.class_dict.iloc[i]['g'], self.class_dict.iloc[i]['b']]
            cmap = np.all(np.equal(mask, rgb_class), axis=-1)
            # im = Image.fromarray(np.uint8(cmap*255))
            # im.show()
            mask_machine.append(cmap)
        mask_machine = np.stack(mask_machine, axis=-1)

        mask_machine = torch.argmax(T.ToTensor()(mask_machine*1), 0)
        return {"img": img,
                "mask_machine": mask_machine}
