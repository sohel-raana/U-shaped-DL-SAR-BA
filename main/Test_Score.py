import os
import sys
# import wandb
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
import seaborn as sns
# Add your code directory to system path
sys.path.append('/share/wildfire-3/Sohel/Model/Code/UNet')

from sklearn.model_selection import train_test_split

directory = '/share/wildfire-3/Sohel/Model/Data/Score/Models_30_1_Trial22/WithZeroMask/BCE_Final_NoAug_blc/MinMax_Final/T_F/'
os.makedirs(directory, exist_ok=True)
os.chdir(directory)

# Import custom modules
from Metrics import calculate_accuracy, DiceLoss, DICE_BCE_Loss
from DataLoad import UNetDataset, UNetDataset_1
from DataLoad_2 import UNetDataset_2

from u_net import build_unet
from attention_unet import attention_unet
from res_attn_unet import AttentionResUNet
from ExtendedNestedUNet_1 import NestedUNetHeTrans, NestedUnet_DeepSup
from UNet3Plus import UNet_3Plus_DeepSup

# Set up device and environment
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
torch.cuda.empty_cache()

# Set random seeds
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Define model architectures
# import segmentation_models_pytorch as smp

# Define model architectures
model_architectures = [
    ('Unet_Res50', lambda c: smp.create_model(
    "unet", encoder_name='resnet50', encoder_weights=None, in_channels=c, classes=1, activation= "sigmoid")
    ),
    ('U_Net', lambda c: build_unet(in_channels=c)),
    ('AttU_Net', lambda c: attention_unet(in_channels=c)),
    ('AttentionResUNet', lambda c: AttentionResUNet(c, 1)),
    ('NestedUNet_He', lambda c: NestedUNetHeTrans(c, 1)),
    ('NestedUnet_DeepSup', lambda c: NestedUnet_DeepSup(c, 1)),
    ('UNet_3Plus_DeepSup', lambda c: UNet_3Plus_DeepSup(c, 1)),
)
]

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score

def calculate_metrics(predictions, ground_truth):
    """
    Computes Accuracy, Precision, Recall, F1 Score, and IoU using sklearn.

    Arguments:
        predictions (numpy array or tensor): Predicted binary values (0 or 1).
        ground_truth (numpy array or tensor): Ground truth binary labels (0 or 1).

    Returns:
        accuracy, precision, recall, f1_score, iou: Computed metrics.
    """
    # Flatten arrays to ensure they are 1D
    predictions = predictions.flatten()
    ground_truth = ground_truth.flatten()

    # Compute metrics using sklearn
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions, zero_division=0)
    recall = recall_score(ground_truth, predictions, zero_division=0)
    f1 = f1_score(ground_truth, predictions, zero_division=0)
    iou = jaccard_score(ground_truth, predictions, zero_division=0)  # IoU is also known as the Jaccard score

    return accuracy, precision, recall, f1, iou



# Function to evaluate a model on a given DataLoader
def evaluate_model(loader, data_split):
    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for images, masks in tqdm(loader, total=len(loader), desc=f"Evaluating {data_split} Data"):
            images, masks = images.to(device), masks.to(device)

            # Generate predictions based on model type
            if model_name == 'NestedUnet_DeepSup':
                predictions, _, _, _, _ = model(images)
            elif model_name == 'UNet_3Plus_DeepSup':
                predictions, _, _, _, _ = model(images)
            else:
                predictions = model(images)

            # Convert predictions to binary (0 or 1) using a threshold
            predictions = (predictions > 0.5).float()

            # Accumulate predictions and ground truth
            all_predictions.append(predictions.cpu().numpy())
            all_ground_truths.append(masks.cpu().numpy())

    # Flatten the accumulated predictions and ground truth arrays
    all_predictions = np.concatenate(all_predictions, axis=0).astype(int).flatten()
    all_ground_truths = np.concatenate(all_ground_truths, axis=0).astype(int).flatten()

    # Calculate confusion matrix and metrics for the entire dataset
    accuracy, precision, recall, f1, iou = calculate_metrics(all_predictions, all_ground_truths)

    # Save metrics to a list for later use
    metrics_data.append({
        'Model': model_name,
        'Data Split': data_split,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'mIoU': iou
    })

    # Save metrics to a CSV file
    metrics_output_path = join(directory, f'{model_name}_{data_split}_metrics.csv')
    pd.DataFrame([metrics_data[-1]]).to_csv(metrics_output_path, index=False)
    print(f"Metrics saved to {metrics_output_path}")


# Training parameters , 'Combo', 'RF_Log'
mod_types = ['Log'] #, 'Log', 'Ind', 'Combo'
 
size = 256
lr = 0.001
epochs = 80

# Initialize a list to store metrics for all models and mod_types
all_metrics_data = []

channels_map = {
    'Log': 3,
    'Ind': 2,
    'Combo': 5,
    'RF_Log': 4,
    'Log_2': 2,
    'Combo_2': 4
}

for mod_type in mod_types:
    chnl = channels_map.get(mod_type, 1)
    print(f"Model type: {mod_type}, Channels: {chnl}")
    Loss='BCE'
    # Paths to images and masks
    imgs_patch = f'/share/wildfire-3/Sohel/Model/Data/Data/Main_2/{mod_type}/Patch_256_4/Data/'
    msks_patch = f'/share/wildfire-3/Sohel/Model/Data/Data/Main_2/{mod_type}/Patch_256_4/Mask/' 

    image_paths_patch = sorted(list(iglob(join(imgs_patch, '*.npy'), recursive=True)))
    mask_paths_patch = sorted(list(iglob(join(msks_patch, '*.npy'), recursive=True)))

    # Function to filter filenames
    def has_two_numbers(filename):
        pattern = r'\d+_\d+\.npy$'
        return re.search(pattern, filename) is not None

    # Filter out specific filenames
    filtered_image_paths = [img for img in image_paths_patch if not has_two_numbers(os.path.basename(img))]
    filtered_mask_paths = [mask for mask in mask_paths_patch if not has_two_numbers(os.path.basename(mask))]

    # Train/test split
    train_filenames = ['laTorre','Avila','Pedrogao',
    'Monchique','Marinha','Folgoso', 'Drosopigi','Ermida',
    'Bejis','Makrimalli','Moros','LaDrova',
    'Pinofranquendo','Evia']
    
    val_filenames = ['SierraPatras',]

    test_filenames = ['Patras','BaraodeSaoJoao',
    'AncientOlympia','Jubrique','VilaDeRei','Sierra',
    'Salakos','VilarinhodeSamarda',
    'Kalamos'] #Main one used overall

    # test_filenames = ['BaraodeSaoJoao',
    # 'AncientOlympia','Jubrique','VilaDeRei',
    # 'Salakos','VilarinhodeSamarda',
    # 'Kalamos']

    train_images_filtered = [img for img in filtered_image_paths if os.path.basename(img).split('_')[0] in train_filenames]
    train_masks_filtered = [mask for mask in filtered_mask_paths if os.path.basename(mask).split('_')[0] in train_filenames]
 
    test_masks_filtered = [mask for mask in filtered_mask_paths if os.path.basename(mask).split('_')[0] in test_filenames]
    test_images_filtered = [img for img in filtered_image_paths if os.path.basename(img).split('_')[0] in test_filenames]
    
    metric_data = [img for img in filtered_image_paths if os.path.basename(img).split('_')[0] in train_filenames]
    metric_mask = [mask for mask in filtered_mask_paths if os.path.basename(mask).split('_')[0] in train_filenames]

    print("Number of train images:", len(train_images_filtered))
    print("Number of test images:", len(test_images_filtered))
    print("Number of train masks:", len(train_masks_filtered))
    print("Number of test masks:", len(test_masks_filtered))

    # Load images and masks
    images = [np.load(img) for img in train_images_filtered]
    masks = [np.load(mask) for mask in train_masks_filtered]

    # Compute fire pixel ratio per image
    fire_ratios = [np.sum(mask) / mask.size for mask in masks]  # % of fire pixels per image

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
        test_size=0.2,  # 80% train, 20% test
        stratify=stratify_labels,  # Ensures balance across fire severity bins
        random_state=42
    )

    # Create datasets and data loaders with normalization
    train_dataset = UNetDataset_2(train_images, train_masks, discard_channel=None, augment=True)
    test_dataset = UNetDataset_1(test_images_filtered, test_masks_filtered, discard_channel=None)
    val_dataset = UNetDataset_2(test_images, test_masks,discard_channel=None)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Create a dictionary to store metrics
    metrics_data = []

    for model_name, model_constructor in model_architectures:
        print(f"Processing model: {model_name} for dataset {mod_type}")
        model = model_constructor(chnl).to(device)
        # model_weights_path = f'/share/wildfire-3/Sohel/Model/Data/Checkpoints/UKAN_Models/FireSegmentation/UKAN_Fire_Segmentation/UKAN_Log_256_0.001_epoch43_best.pth'
        model_weights_path = '/share/wildfire-3/Sohel/Model/Data/Checkpoints/Models_30_1_Trial22/WithZeroMask/BCE_Final_NoAug_blc/MinMax_1/Trial-x-2/Log_Final_256_0.001_SwinUNet_BCE.pth'

        model.load_state_dict(torch.load(model_weights_path, map_location=device))
        model.eval()

        # Evaluate on train and test loaders
        # evaluate_model(train_loader, 'Train')
        evaluate_model(test_loader, 'Test')
        # evaluate_model(val_loader, 'Val')

        # Append mod_type to metrics data
        for metric in metrics_data:
            metric['Mod Type'] = mod_type

        # Aggregate into all_metrics_data
        all_metrics_data.extend(metrics_data)

        # Save metrics to a CSV file
        metrics_df = pd.DataFrame(metrics_data)
        print(metrics_df)
        output_path = join(directory, f'{mod_type}_model_metric-RFDI-DIce-train.csv')
        # metrics_df.to_csv(output_path, index=False)

        print(f"Metrics saved to {output_path}")

# Save all metrics to a single CSV file
all_metrics_df = pd.DataFrame(all_metrics_data)
output_path = join(directory, 'all_model_metrics-RVI-Focal.csv')
all_metrics_df.to_csv(output_path, index=False)

print(f"All metrics saved to {output_path}")
