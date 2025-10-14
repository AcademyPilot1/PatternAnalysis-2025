# Note: I am mainly working in google colab with jupyter notebooks, so these files
# are essentially copies of each of the colab cells in order

# also assuming the 'modules.py' file is for the model architecture

# Various import statements
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np


# ---- ConvNeXt Block ----
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)))

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)       # NCHW → NHWC
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)       # NHWC → NCHW
        return shortcut + x
    
    
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:  # channels_first
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            return self.weight[:, None, None] * x + self.bias[:, None, None]    
        
        
# ---- Tiny ConvNeXt Classifier ----
class TinyConvNeXt(nn.Module):
    def __init__(self, in_chans=3, num_classes=2, dim=96):
        super().__init__()
        # "stem" - patchify
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dim, kernel_size=4, stride=4),
            LayerNorm(dim, eps=1e-6, data_format="channels_first")
        )
        # a single ConvNeXt block
        self.block = ConvNeXtBlock(dim)
        # global pooling + head
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block(x)
        x = x.mean([-2, -1])  # Global Average Pool
        x = self.norm(x)
        x = self.head(x)
        return x