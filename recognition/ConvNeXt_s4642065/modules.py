# Note: I am mainly working in google colab with jupyter notebooks, so these files
# are essentially copies of each of the colab cells in order

# also assuming the 'modules.py' file is for the model architecture

# Various import statements
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time as time
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from timm.layers import trunc_normal_, DropPath



class DropPath(nn.Module):
    """Stochastic Depth (Drop Path) for regularization"""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class SmallBlock(nn.Module):
    """ConvNeXt block with 5x5 depthwise conv and stochastic depth"""
    def __init__(self, dim, layer_scale_init_value=1e-5, drop_path=0.0):
        super().__init__()
        # 5x5 depthwise convolution
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pw1 = nn.Linear(dim, 4*dim)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(4*dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma = None

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)                          # N,C,H,W
        x = x.permute(0, 2, 3, 1)                   # N,H,W,C
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)                   # N,C,H,W
        return shortcut + self.drop_path(x)


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)
    def forward(self, x):
        # N, C, H, W → N, H, W, C → N, C, H, W
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class ConvNeXt(nn.Module):
    def __init__(self, in_chans=1, num_classes=2,
                 depths=(2, 2, 6, 2), dims=(48, 96, 192, 384),
                 layer_scale_init_value=1e-5, drop_path_rate=0.1):
        super().__init__()
        assert len(depths) == 4 and len(dims) == 4

        self.dropout = nn.Dropout(p=0.2)

        # Stem
        self.downsamples = nn.ModuleList()

        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0])
        )
        self.downsamples.append(stem)

        # Downsampling layers between stages
        for i in range(3):
            self.downsamples.append(
                nn.Sequential(
                    nn.GroupNorm(1, dims[i]),  # acts like LayerNorm for channels_first
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
                )
            )

        # Calculate stochastic depth rates (linearly increasing)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build stages with progressive drop path
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(4):
            blocks = []
            for _ in range(depths[i]):
                blocks.append(SmallBlock(dims[i],
                                        layer_scale_init_value=layer_scale_init_value,
                                        drop_path=dp_rates[cur]))
                cur += 1
            self.stages.append(nn.Sequential(*blocks))

        # Final norm and head
        self.final_norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        # x: N, C, H, W  (C=1)
        for i in range(4):
            x = self.downsamples[i](x)
            x = self.stages[i](x)
        # Global average pooling
        x = x.mean([-2, -1])            # N, C
        x = self.final_norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.dropout(x)
        x = self.head(x)
        return x


#
#
## RESULTS
## Epochs:20, Final loss: 0.0099, Accuracy: 63.78 (0.5 normalisation) (32 batch size) (epoch time 125s)
## ok so next one was horrible, 49.18 with scheduler, 16 batch, custom transformer
##train_transform = transforms.Compose([
##    transforms.Grayscale(num_output_channels=3),
##    transforms.Resize((224, 224)),
##    transforms.RandomHorizontalFlip(p=0.5),
##    transforms.RandomRotation(degrees=10),
#    transforms.ColorJitter(brightness=0.2, contrast=0.2),
#    transforms.ToTensor(),
#    transforms.Normalize(mean=mean.tolist(), std=std.tolist())
#])

#Epoch 20, Loss: 0.1063, Time: 162.90509629249573
#Finished Training on 21520 images in 3225.07324385643 seconds

#Beginning Testing:
#Test Accuracy: 67.84%
#Finished Testing in 45.61187219619751 seconds

# best result yet acheived on smaller convnext model = MiniConvNeXt(in_chans=1, num_classes=2,
#                 depths=(1,1,2,1), dims=(32,64,128,256)).to(device)
# after 100 epochs, final loss 0.0653 (best 0.606) accuracy 75.28%
# this was after reverting to layernorm and kernel size = 3
# model trained in 4922 seconds

# ok improvement with dropout 0.2 added to model forward and new transforms
# after 100 epochs, final loss 0.0388 (best 0.0350) accuracy only marginally
# better at 76.1%