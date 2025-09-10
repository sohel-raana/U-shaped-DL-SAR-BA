import os
import numpy as np
import torch
from torch.utils.data import Dataset

import torchvision.transforms as transforms

class UNetDataset_1(Dataset):
    def __init__(self, image_files, mask_files, mean=None, std=None, discard_channel=None):
        """
        Args:
            image_files (list): List of paths to image files.
            mask_files (list): List of paths to mask files.
            mean (list, optional): List of mean values for each channel. If None, no normalization is applied.
            std (list, optional): List of standard deviation values for each channel. If None, no normalization is applied.
            discard_channel (int, optional): Index of the channel to discard. If None, all channels are kept.
        """
        self.image_files = image_files
        self.mask_files = mask_files
        self.discard_channel = discard_channel  # Channel to discard
        if mean is not None and std is not None:
            self.normalize = transforms.Normalize(mean=mean, std=std)  # Use torchvision normalization
        else:
            self.normalize = None  # No normalization

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load images and masks
        image = np.load(self.image_files[idx])
        mask = np.load(self.mask_files[idx])

        # Discard specified channel if defined
        if self.discard_channel is not None:
            image = np.delete(image, self.discard_channel, axis=-1)  # Remove the specified channel (H, W, C -> H, W, C-1)

        # Convert to PyTorch tensors and adjust shape
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Convert to (C, H, W)
        if mask.ndim == 2:  # If mask has no third dimension
            mask = np.expand_dims(mask, axis=-1)
        mask = torch.tensor(mask, dtype=torch.float32).permute(2, 0, 1)  # Convert to (C, H, W)

        # Apply normalization if defined
        if self.normalize:
            image = self.normalize(image)
        
        return image, mask

import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms

class UNetDataset_2(Dataset):
    def __init__(self, images, masks, mean=None, std=None, discard_channel=None):
        """
        Args:
            images (list of np.ndarray): List of loaded image arrays.
            masks (list of np.ndarray): List of loaded mask arrays.
            mean (list, optional): List of mean values for each channel.
            std (list, optional): List of standard deviation values for each channel.
            discard_channel (int, optional): Index of the channel to discard (if any).
        """
        self.images = images
        self.masks = masks
        self.discard_channel = discard_channel  # Channel to discard
        
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

        # Apply normalization if defined
        if self.normalize:
            image = self.normalize(image)
        
        return image, mask

class UNetDataset(Dataset):
    def __init__(self, image_files, mask_files, dataset_type, transform=None):
        self.image_files = image_files
        self.mask_files = mask_files
        self.dataset_type = dataset_type
        self.transform = transform
        
        # Define mean and std for each dataset type
        self.normalization_params = {
            "DPSVI_No": {
                "mean": torch.tensor([-0.011603, -0.031644], dtype=torch.float32),
                "std": torch.tensor([0.068066, 0.139218], dtype=torch.float32)
            },
            "Log_No": {
                "mean": torch.tensor([-0.006801, 0.017755], dtype=torch.float32),
                "std": torch.tensor([0.172949, 0.174581], dtype=torch.float32)
            },
            "Mixed_No": {
                "mean": torch.tensor([-0.011603, -0.031644, -0.006801, 0.017755], dtype=torch.float32),
                "std": torch.tensor([0.068066, 0.139218, 0.172949, 0.174581], dtype=torch.float32)
            }
        }
        
        if self.dataset_type not in self.normalization_params:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load images and masks
        image = np.load(self.image_files[idx])
        mask = np.load(self.mask_files[idx])

        # Convert to PyTorch tensors and adjust shape
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        if mask.ndim == 2:  # If mask has no third dimension
            mask = np.expand_dims(mask, axis=-1)
        mask = torch.tensor(mask, dtype=torch.float32).permute(2, 0, 1)
        
        # Get mean and std for the dataset type
        normalization = self.normalization_params[self.dataset_type]
        mean = normalization["mean"].view(-1, 1, 1)
        std = normalization["std"].view(-1, 1, 1)

        # Normalize the image
        image = (image - mean) / std

        # Apply transform
        if self.transform:
            image, mask = self.transform(image, mask)
            
        return image, mask

class UNetDataset_Chnl(Dataset):
    def __init__(self, image_files, mask_files, dataset_type, transform=None, selected_channels=None):
        """
        Args:
            image_files (list): List of paths to image files.
            mask_files (list): List of paths to mask files.
            dataset_type (str): The type of dataset (e.g., "DPSVI_No", "Log_No", "Mixed_No").
            transform (callable, optional): A function/transform to apply to the images and masks.
            selected_channels (list, optional): List of channel indices to use (e.g., [0], [1], [0, 1]).
        """
        self.image_files = image_files
        self.mask_files = mask_files
        self.dataset_type = dataset_type
        self.transform = transform
        self.selected_channels = selected_channels  # New parameter
        
        # Define mean and std for each dataset type
        self.normalization_params = {
            "DPSVI_No": {
                "mean": torch.tensor([-0.011603, -0.031644], dtype=torch.float32),
                "std": torch.tensor([0.068066, 0.139218], dtype=torch.float32)
            },
            "Log_No": {
                "mean": torch.tensor([-0.006801, 0.017755], dtype=torch.float32),
                "std": torch.tensor([0.172949, 0.174581], dtype=torch.float32)
            },
            "Mixed_No": {
                "mean": torch.tensor([-0.011603, -0.031644, -0.006801, 0.017755], dtype=torch.float32),
                "std": torch.tensor([0.068066, 0.139218, 0.172949, 0.174581], dtype=torch.float32)
            }
        }
        
        if self.dataset_type not in self.normalization_params:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load images and masks
        image = np.load(self.image_files[idx])
        mask = np.load(self.mask_files[idx])

        # Convert to PyTorch tensors and adjust shape
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        if mask.ndim == 2:  # If mask has no third dimension
            mask = np.expand_dims(mask, axis=-1)
        mask = torch.tensor(mask, dtype=torch.float32).permute(2, 0, 1)
        
        # Select specific channels if specified
        if self.selected_channels is not None:
            image = image[1, :, :]  # Select specific channels

        # Get mean and std for the dataset type
        normalization = self.normalization_params[self.dataset_type]
        mean = normalization["mean"].view(-1, 1, 1)
        std = normalization["std"].view(-1, 1, 1)

        # Adjust mean and std for selected channels
        if self.selected_channels is not None:
            mean = mean[self.selected_channels, :, :]
            std = std[self.selected_channels, :, :]

        # Normalize the image
        image = (image - mean) / std

        # Apply transform
        if self.transform:
            image, mask = self.transform(image, mask)
            
        return image, mask

class UNetDataset_new(Dataset):
    def __init__(self, image_files, mask_files):
        self.image_files = image_files
        self.mask_files = mask_files
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load images and masks
        image = np.load(self.image_files[idx])
        mask = np.load(self.mask_files[idx])

        # Convert to PyTorch tensors and adjust shape
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        if mask.ndim == 2:  # If mask has no third dimension
            mask = np.expand_dims(mask, axis=-1)
        mask = torch.tensor(mask, dtype=torch.float32).permute(2, 0, 1)
        
        return image, mask
        
class UNetDatasetMinMax(Dataset):
    def __init__(self, image_files, mask_files, norm_mode=None, transform=None):
        self.image_files = image_files
        self.mask_files = mask_files
        self.norm_mode = norm_mode
        self.transform = transform

        # Define quantiles for different normalization modes
        self.quantiles = {
            "DPSVI_No": {
                "min": torch.tensor([-0.178672, -0.474185], dtype=torch.float32),
                "max": torch.tensor([0.128506, 0.305733], dtype=torch.float32),
            },
            "Log_No": {
                "min": torch.tensor([-0.474391, -0.394220], dtype=torch.float32),
                "max": torch.tensor([0.512315, 0.576090], dtype=torch.float32),
            },
            "Mixed_No": {
                "min": torch.tensor([-0.176099, -0.466889, -0.467547, -0.387967], dtype=torch.float32),
                "max": torch.tensor([0.125395, 0.298585, 0.506425, 0.568190], dtype=torch.float32),
            }
        }

        if self.norm_mode not in self.quantiles:
            raise ValueError(f"Unknown normalization mode: {self.norm_mode}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load images and masks
        image = np.load(self.image_files[idx])
        mask = np.load(self.mask_files[idx])

        # Convert to PyTorch tensors and adjust shape
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        if mask.ndim == 2:  # If mask has no third dimension
            mask = np.expand_dims(mask, axis=-1)
        mask = torch.tensor(mask, dtype=torch.float32).permute(2, 0, 1)  # (H, W) -> (1, H, W)

        # Get quantile min and max values for normalization
        normalization = self.quantiles[self.norm_mode]
        min_vals = normalization["min"].view(-1, 1, 1)  # Reshape for broadcasting
        max_vals = normalization["max"].view(-1, 1, 1)  # Reshape for broadcasting

        # Normalize each channel using the quantiles
        image = (image - min_vals) / (max_vals - min_vals)
        image = torch.clamp(image, 0, 1)  # Clip to [0, 1]

        # Apply transform
        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask
