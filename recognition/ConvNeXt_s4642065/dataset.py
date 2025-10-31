# dataset.py
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
import os

class ADNIDataLoader:
    """
    Custom data loader for the ADNI dataset.
    Handles dataset loading, preprocessing transforms, normalization, and validation split.
    """

    def __init__(self, data_dir="data/ADNI/AD_NC", batch_size=128, num_workers=2,
                 val_split=0.2, compute_norm=True):
        """
        Initialize the ADNI data loader.

        Args:
            data_dir (str): Base directory containing 'train' and 'test' folders.
            batch_size (int): Batch size for data loaders.
            num_workers (int): Number of workers for data loading.
            val_split (float): Fraction of training data to use for validation.
            compute_norm (bool): Whether to compute mean/std from training set.
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

        # Default mean/std placeholders
        self.mean = [0.5]
        self.std = [0.5]

        # Compute normalization from training set if required
        if compute_norm:
            self.mean, self.std = self._compute_normalization_params()

        # Define transforms
        self.train_transform = self._transform_train()
        self.test_transform = self._transform_test()

    # ------------------------------------------------------------------
    def _compute_normalization_params(self):
        """Compute dataset mean and std from training images."""
        print("Computing dataset normalization parameters...")

        temp_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        temp_dataset = datasets.ImageFolder(root=f"{self.data_dir}/train", transform=temp_transform)
        temp_loader = DataLoader(temp_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        mean, std, total = 0.0, 0.0, 0
        for images, _ in tqdm(temp_loader, desc="Calculating mean/std"):
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            total += batch_samples

        mean /= total
        std /= total
        print(f" Computed mean={mean.tolist()}, std={std.tolist()}")
        return mean.tolist(), std.tolist()

    # ------------------------------------------------------------------
    def _transform_train(self):
        """Define training transforms."""
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
        """Define testing/validation transforms."""
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    # ------------------------------------------------------------------
    def _split_train_val(self, dataset):
        """Split a dataset into training and validation subsets."""
        total_size = len(dataset)
        val_size = int(total_size * self.val_split)
        train_size = total_size - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        print(f"Split training data into {train_size} train and {val_size} validation samples.")
        return train_dataset, val_dataset

    # ------------------------------------------------------------------
    def load(self):
        """
        Load train, validation, and test datasets and return their DataLoaders.

        Returns:
            (train_loader, val_loader, test_loader)
        """
        # Full train dataset (to be split)
        full_train_dataset = datasets.ImageFolder(root=f"{self.data_dir}/train", transform=self.train_transform)
        train_dataset, val_dataset = self._split_train_val(full_train_dataset)

        # Test dataset
        test_dataset = datasets.ImageFolder(root=f"{self.data_dir}/test", transform=self.test_transform)

        # Dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        print(f"Loaded {len(train_dataset)} train, {len(val_dataset)} val, and {len(test_dataset)} test samples.")
        return train_loader, val_loader, test_loader
