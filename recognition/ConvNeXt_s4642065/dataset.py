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
import copy
from timm.layers import trunc_normal_, DropPath
from tqdm import tqdm


class ADNIDataLoader():
    """
    Customer data loader for loading ADNI data set
    Provides image transforms and normalisation
    
    """
    
    def __init__(self, data_dir="data/ADNI/AD_NC", batch_size=128, num_workers=2, compute_norm=True):
        """
        Initialize the ADNI data loader.

        Args:
            data_dir (str): Base directory containing 'train' and 'test' folders.
            batch_size (int): Batch size for data loaders.
            num_workers (int): Number of workers for data loading.
            compute_norm (bool): Whether to compute mean/std from training set.
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Default mean/std placeholders
        self.mean = [0.5]
        self.std = [0.5]

        # Compute normalization from training set if required
        if compute_norm:
            self.mean, self.std = self._compute_normalization_params()

        # Set up transforms
        self.train_transform = self._transform_train()
        self.test_transform = self._transform_test()
    
    
    def load(self):
        """
        Load training and test datasets and return their DataLoaders.

        Returns:
            (train_loader, test_loader): tuple of PyTorch DataLoaders
        """
        train_dataset = datasets.ImageFolder(root=f"{self.data_dir}/train", transform=self.train_transform)
        test_dataset = datasets.ImageFolder(root=f"{self.data_dir}/test", transform=self.test_transform)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        print(f"Loaded {len(train_dataset)} training and {len(test_dataset)} testing samples.")
        return train_loader, test_loader
        
        
    def _compute_normalization_params(self):
        """
        Compute dataset mean and std from the training images.
        """
        print("Computing dataset normalization parameters...")
        temp_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        temp_dataset = datasets.ImageFolder(root=f"{self.data_dir}/train", transform=temp_transform)
        temp_loader = DataLoader(temp_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        mean = 0.0
        std = 0.0
        total = 0

        for images, _ in tqdm(temp_loader, desc="Calculating mean/std"):
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)  # (B, C, H*W)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            total += batch_samples

        mean /= total
        std /= total
        print(f" → Computed mean={mean.tolist()}, std={std.tolist()}")
        return mean.tolist(), std.tolist()      
            
    
    def _transform_train(self):
        """
        Define the training data transformations.
        """
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(12),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def _transform_test(self):
        """
        Define the test data transformations.
        """
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    
