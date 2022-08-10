from torch import nn
import torch


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, num_classes):
        super(NeuralNetwork, self).__init__()
        self.backbone = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5)
        self.backbone.n_classes = num_classes
        for params in self.backbone.parameters():
            params.require_grad = False
        self.backbone.outc = nn.Sequential(
            nn.Conv2d(64, num_classes, kernel_size=(1, 1), stride=(1, 1))
        )

    def forward(self, x):
        x = self.backbone(x)
        return x

    def predict(self, img):
        self.backbone.eval()
        with torch.no_grad():
            pred = self.backbone(img)
            return pred
