import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import matplotlib.pyplot as plt
from glob import iglob
from os.path import join
import re

# Add your code directory to system path
sys.path.append('/share/wildfire-3/Sohel/Model/Code/UNet')

# Set up directories
directory = '/share/wildfire-3/Sohel/Model/Data/Checkpoints/Models_30_1_Trial22/WithZeroMask/BCE_Final_NoAug_blc/MinMax_1/Trial-x-2/'
os.makedirs(directory, exist_ok=True)
os.chdir(directory)

# Import custom modules
from Metrics import calculate_accuracy, FocalLoss, DiceLoss, FocalDiceLoss
from DataLoad_2 import UNetDataset_2
from AttentionUNet import AttU_Net, U_Net

from u_net import build_unet
from attention_unet import attention_unet
from res_attn_unet import AttentionResUNet
from ExtendedNestedUNet_1 import NestedUNetHeTrans, NestedUnet_DeepSup
from UNet3Plus import UNet_3Plus_DeepSup

# Set up device and environment
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
torch.cuda.empty_cache()

# Set random seeds for reproducibility
def set_random_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Define model architectures
# import segmentation_models_pytorch as smp

# Define model architectures
def get_model_architectures():
    return [
        ('Unet_Res50', lambda c: smp.create_model(
            "unet", encoder_name='resnet50', encoder_weights=None, in_channels=c, classes=1, activation= "sigmoid")
        ),
        ('U_Net', lambda c: build_unet(in_channels=c)),
        ('AttU_Net', lambda c: attention_unet(in_channels=c)),
        ('AttentionResUNet', lambda c: AttentionResUNet(c, 1)),
        ('NestedUNet_He', lambda c: NestedUNetHeTrans(c, 1)),
        ('NestedUnet_DeepSup', lambda c: NestedUnet_DeepSup(c, 1)),
        ('UNet_3Plus_DeepSup', lambda c: UNet_3Plus_DeepSup(c, 1)),
    ]

lr = 0.001
batch_size = 8

# Function to filter filenames
def has_two_numbers(filename):
    pattern = r'\d+_\d+\.npy$'
    return re.search(pattern, filename) is not None

from sklearn.model_selection import train_test_split

def prepare_data(mod_type, size):
    # Paths to images and masks
    imgs_patch = f'/share/wildfire-3/Sohel/Model/Data/Data/Main_2/{mod_type}/Patch_256_4/Data/'
    msks_patch = f'/share/wildfire-3/Sohel/Model/Data/Data/Main_2/{mod_type}/Patch_256_4/Mask/' 

    image_paths_patch = sorted(list(iglob(join(imgs_patch, '*.npy'), recursive=True)))
    mask_paths_patch = sorted(list(iglob(join(msks_patch, '*.npy'), recursive=True)))

    # Filter out specific filenames
    filtered_image_paths = [img for img in image_paths_patch if not has_two_numbers(os.path.basename(img))]
    filtered_mask_paths = [mask for mask in mask_paths_patch if not has_two_numbers(os.path.basename(mask))]

    # Train/test split
    train_filenames = ['laTorre','Avila','Pedrogao',
    'Monchique','Marinha','Folgoso', 'Drosopigi','Ermida',
    'Bejis','Makrimalli','Moros','LaDrova',
    'Pinofranquendo','Evia','Ladrillar']
    
    val_filenames = []

    test_filenames = ['Patras','BaraodeSaoJoao',
    'AncientOlympia','Jubrique','VilaDeRei','Sierra',
    'Salakos','VilarinhodeSamarda',
    'Kalamos']

    train_images_filtered = [img for img in filtered_image_paths if os.path.basename(img).split('_')[0] in train_filenames]
    train_masks_filtered = [mask for mask in filtered_mask_paths if os.path.basename(mask).split('_')[0] in train_filenames]

    # Load images and masks
    images = [np.load(img) for img in train_images_filtered]
    masks = [np.load(mask) for mask in train_masks_filtered]

    # Compute fire pixel ratio per image
    fire_ratios = [np.sum(mask) / mask.size for mask in masks]  # % of fire pixels per image
    print(fire_ratios)
    # Create stratification bins
    def assign_bin(ratio):
        if ratio == 0.0:
            return 0  # No Fire
        elif 0.0 < ratio <= 0.3:
            return 1  # Low Fire
        elif 0.3 < ratio <= 0.:
            return 2  # Medium6 Fire
        else:
            return 3  # High Fire

    stratify_labels = [assign_bin(ratio) for ratio in fire_ratios]

    # Perform stratified train-test split ensuring balance across bins
    train_images, test_images, train_masks, test_masks = train_test_split(
        images, masks, 
        test_size=0.20,  # 80% train, 20% test
        stratify=stratify_labels,  # Ensures balance across fire severity bins
        random_state=42
    )

    print(train_images)
    return train_images, test_images, train_masks, test_masks
    
# Modify train_model to include this function
def train_model(model, train_loader, test_loader, model_name, mod_type, size, lr, epochs, loss_name, loss_function):
    
    # Move model to device
    model = model.to(device)
    model = model.cuda()

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=4
    )

    # CosineAnnealingLR expects T_max (number of epochs for a cycle) and optionally eta_min (minimum LR)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=epochs, eta_min=1e-6
    # )

    # Training parameters
    best_val_loss = float('inf')
    no_improvement_count = 0
    train_losses, val_losses = [], []
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss, train_accuracy = 0, 0
        
        # Training phase
        for images, masks in tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1} Training"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            
            outputs_train = model(images)
            
            # Handle different model architectures with multiple outputs
            if isinstance(outputs_train, tuple):
                loss = 0
                for output in outputs_train:
                    loss += loss_function(output, masks)
                loss /= len(outputs_train)
                train_accuracy += calculate_accuracy(outputs_train[0], masks)
            else:
                loss = loss_function(outputs_train, masks)
                train_accuracy += calculate_accuracy(outputs_train, masks)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        # Average train metrics
        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)
        train_losses.append(train_loss)
        
        print(f"Epoch {epoch + 1} - Training Loss: {train_loss:.6f}, Training Accuracy: {train_accuracy:.6f}")

        # Validation phase
        model.eval()
        val_loss, val_accuracy = 0, 0
        with torch.no_grad():
            for images, masks in tqdm(test_loader, total=len(test_loader), desc=f"Epoch {epoch + 1} Validation"):
                images, masks = images.to(device), masks.to(device)
                outputs_val = model(images)
                
                # Handle different model architectures with multiple outputs
                if isinstance(outputs_val, tuple):
                    batch_val_loss = 0
                    for output in outputs_val:
                        batch_val_loss += loss_function(output, masks)
                    batch_val_loss /= len(outputs_val)
                    val_accuracy += calculate_accuracy(outputs_val[0], masks)
                else:
                    batch_val_loss = loss_function(outputs_val, masks)
                    val_accuracy += calculate_accuracy(outputs_val, masks)
                
                val_loss += batch_val_loss.item()

        # Average validation metrics
        val_loss /= len(test_loader)
        val_accuracy /= len(test_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1} - Validation Loss: {val_loss:.6f}, Validation Accuracy: {val_accuracy:.6f}")

        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_count = 0
            model_filename = f'{mod_type}_Final_{size}_{lr}_{model_name}_{loss_name}.pth'
            torch.save(model.state_dict(), model_filename)
            print(f"New best model saved at: {model_filename}")
        else:
            no_improvement_count += 1
        
        if no_improvement_count >= 10:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break
        
        # Step the scheduler
        scheduler.step(val_loss)

    # Plot and save loss curves
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss for {model_name} with {mod_type}')
    plt.legend()
    plt.savefig(f'{model_filename}_loss.png', dpi=300, bbox_inches='tight')
    plt.close()



# Set random seeds
set_random_seeds()

# Training configurations
size = 256
epochs = 70

mod_types = ['Log'] #,'Log_2','Ind','Combo'
model_architectures = get_model_architectures()


loss_name = 'BCE'
loss_function = nn.BCELoss()

# Mapping mod_type to the number of channels
channels_map = {
    'Log': 3,
    'Ind': 2,
    'Combo': 5,
    # 'RFDI': 1,
    'Log_2': 2,
    # 'Combo_2': 4
}

for mod_type in mod_types:
    # Get the number of channels based on the mod_type
    chnl = channels_map.get(mod_type, 1)  # Default to 1 if mod_type is not in the map
    print(f"Model Type: {mod_type}, Channels: {chnl}")

    # Prepare data
    train_images, val_images, train_masks, val_masks = prepare_data(mod_type, size)
    
    print("Number of train images:", len(train_images))
    print("Number of Val images:", len(val_images))
    print("Number of train masks:", len(train_masks))
    print("Number of Val masks:", len(val_masks))

    # Create datasets and data loaders with augmentation for training
    train_dataset = UNetDataset_2(train_images, train_masks, discard_channel=None, augment=True)  # Enable augmentation for training
    val_dataset = UNetDataset_2(val_images, val_masks, discard_channel=None, augment=False)  # No augmentation for validation

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


    # Train each model architecture
    for model_name, model_constructor in model_architectures:
        print(f"Training {model_name} with {mod_type}")
        
        # Initialize model
        model = model_constructor(chnl)
        
        # Train the model
        train_model(model, train_loader, val_loader, model_name, mod_type, size, lr, epochs, loss_name, loss_function)

        print(f"Training complete for {model_name} with {mod_type}.")
