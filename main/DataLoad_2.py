import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms

class UNetDataset_2(Dataset):
    def __init__(self, images, masks, mean=None, std=None, discard_channel=None, augment=False):
        """
        Args:
            images (list of np.ndarray): List of loaded image arrays.
            masks (list of np.ndarray): List of loaded mask arrays.
            mean (list, optional): List of mean values for each channel.
            std (list, optional): List of standard deviation values for each channel.
            discard_channel (int, optional): Index of the channel to discard (if any).
            augment (bool, optional): Whether to apply data augmentation.
        """
        self.images = images
        self.masks = masks
        self.discard_channel = discard_channel  # Channel to discard
        self.augment = augment  # Enable augmentation

        # Define PyTorch augmentations (only when augment=True)
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),  # 50% chance of horizontal flip
            transforms.RandomVerticalFlip(p=0.5)     # 50% chance of vertical flip
        ])

        # Define normalization if mean and std are provided
        if mean is not None and std is not None:
            self.normalize = transforms.Normalize(mean=mean, std=std)
        else:
            self.normalize = None  # No normalization

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and mask (already in-memory NumPy arrays)
        image = self.images[idx]
        mask = self.masks[idx]

        # Discard specified channel if defined
        if self.discard_channel is not None:
            image = np.delete(image, self.discard_channel, axis=-1)  # Remove channel (H, W, C -> H, W, C-1)

        # Ensure image has 3 dimensions (C, H, W) even for grayscale images
        if image.ndim == 2:  # If grayscale (H, W)
            image = np.expand_dims(image, axis=-1)  # Convert to (H, W, 1)

        # Convert NumPy arrays to PyTorch tensors
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # Ensure mask is (1, H, W)

        # Apply augmentation if enabled
        if self.augment:
            stacked = torch.cat([image, mask], dim=0)  # Stack image & mask along channel dimension
            stacked = self.transform(stacked)  # Apply the same augmentation to both
            image, mask = stacked[:-1], stacked[-1:]  # Separate image and mask

        # Apply normalization if defined
        if self.normalize:
            image = self.normalize(image)
        
        return image, mask
