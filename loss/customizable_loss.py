import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from utils.utils import show_clean_noise_image


class Custom_Loss(nn.Module):
    def __init__(self, model: nn.Module, device):
        super().__init__()
        self.model = model
        self.loss = nn.MSELoss()
        self.device = device

    def down_sample(self, image):
        c = image.shape[1]
        # filter1，[[0.5,0],[0,0.5]]
        filter1 = torch.tensor([[[0.5, 0], [0, 0.5]]], dtype=torch.float32, device=self.device)
        filter1 = filter1.repeat(c, 1, 1, 1)
        # filter2, [[0,0.5],[0.5,0]]
        filter2 = torch.tensor([[[0, 0.5], [0.5, 0]]], dtype=torch.float32, device=self.device)
        filter2 = filter2.repeat(c, 1, 1, 1)
        down_sample1 = F.conv2d(image, filter1, stride=2, groups=c)
        down_sample2 = F.conv2d(image, filter2, stride=2, groups=c)

        return down_sample1, down_sample2

    def up_sample(self, down_sample1, down_sample2):
        up_sample1 = F.interpolate(down_sample1, scale_factor=2, mode="bicubic", align_corners=False)
        up_sample2 = F.interpolate(down_sample2, scale_factor=2, mode="bicubic", align_corners=False)
        return up_sample1, up_sample2

    def forward(self, noise_images):
        down1, down2 = self.down_sample(noise_images)
        pred1 = down1 - self.model(down1)
        pred2 = down2 - self.model(down2)
        loss1 = self.loss(pred1, down2)

        noise = noise_images - self.model(noise_images)
        noise1, noise2 = self.down_sample(noise)
        loss2 = 1 / 2 * (self.loss(pred1, noise1) + self.loss(pred2, noise2))

        up_sample1, up_sample2 = self.up_sample(down1, down2)
        up_pred1 = up_sample1 - self.model(up_sample1)
        up_pred2 = up_sample2 - self.model(up_sample2)
        loss3 = self.loss(up_pred1, up_sample2)

        up_noise1, up_noise2 = self.up_sample(noise1, noise2)
        loss4 = 1 / 2 * (self.loss(up_noise1, up_pred1) + self.loss(up_noise2, up_pred2))

        total_loss = loss1 + loss2 + loss3 + loss4

        return total_loss
