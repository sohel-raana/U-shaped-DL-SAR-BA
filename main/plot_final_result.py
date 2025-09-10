import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from glob import iglob
from os.path import join
import re

# Add code directory to system path
sys.path.append('/share/wildfire-3/Sohel/Model/Code/UNet')

# Import custom modules
from DataLoad import UNetDataset_1, UNetDataset_Chnl
from ExtendedNestedUNet import NestedUNetHeTrans, NestedUnet_DeepSup
from UNet3Plus import UNet_3Plus_DeepSup
# Define model architectures
# import segmentation_models_pytorch as smp

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

model_architectures = [
    # ('Unet_Res50', lambda c: smp.create_model(
    #     "unet", encoder_name='resnet50', encoder_weights=None, in_channels=c, classes=1, activation= "sigmoid")
    # ),
    # ('U_Net', lambda c: U_Net(in_channel=c, num_classes=1)),
    # ('AttU_Net', lambda c: AttU_Net(in_channel=c, num_classes=1)),
    # ('AttentionResUNet', lambda c: AttentionResUNet(c, 1)),
    # ('NestedUNet_He', lambda c: NestedUNetHeTrans(c, 1)),
    ('NestedUnet_DeepSup', lambda c: NestedUnet_DeepSup(c, 1)),
    # ('UNet_3Plus_DeepSup', lambda c: UNet_3Plus_DeepSup(c, 1))
]

m = 'Main_2'

def pad_to_match_shape(arr1, arr2):
    """Pad the smaller array to match the shape of the larger array."""
    max_height = max(arr1.shape[0], arr2.shape[0])
    max_width = max(arr1.shape[1], arr2.shape[1])
    
    # Pad arr1
    pad_height1 = max_height - arr1.shape[0]
    pad_width1 = max_width - arr1.shape[1]
    arr1_padded = np.pad(arr1, ((0, pad_height1), (0, pad_width1)), mode='constant')
    
    # Pad arr2
    pad_height2 = max_height - arr2.shape[0]
    pad_width2 = max_width - arr2.shape[1]
    arr2_padded = np.pad(arr2, ((0, pad_height2), (0, pad_width2)), mode='constant')
    
    return arr1_padded, arr2_padded

def pad_to_divisible(image, divisor=32):
    """Pad image to be divisible by divisor"""
    h, w = image.shape[-2:]
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    return torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0)

def crop_to_divisible(image, divisor=32):
    """Crop image so that its dimensions are divisible by a given divisor"""
    h, w = image.shape[-2:]
    new_h = h - (h % divisor)
    new_w = w - (w % divisor)
    return image[..., :new_h, :new_w]

def split_into_patches(image, patch_size=256):
    """Split image into patches"""
    h, w = image.shape[-2:]
    patches = []
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = image[:, i:i + patch_size, j:j + patch_size]
            patches.append((i, j, patch))
    return patches

def merge_patches(patch_predictions, image_shape, patch_size=256):
    """Merge predictions back into full image"""
    h, w = image_shape[-2:]
    merged = np.zeros((h, w), dtype=np.float32)
    for (i, j, patch_pred) in patch_predictions:
        patch_h, patch_w = patch_pred.shape
        merged[i:i + patch_h, j:j + patch_w] = patch_pred
    return merged

def generate_confusion_map(pred_mask, true_mask):
    """Generate colored confusion map"""
    max_h = max(pred_mask.shape[0], true_mask.shape[0])
    max_w = max(pred_mask.shape[1], true_mask.shape[1])
    
    pred_mask = np.pad(pred_mask, ((0, max_h - pred_mask.shape[0]), (0, max_w - pred_mask.shape[1])), mode='constant')
    true_mask = np.pad(true_mask, ((0, max_h - true_mask.shape[0]), (0, max_w - true_mask.shape[1])), mode='constant')
    
    confusion_map = np.zeros(pred_mask.shape + (3,), dtype=np.uint8)
    tp = (pred_mask == 1) & (true_mask == 1)
    tn = (pred_mask == 0) & (true_mask == 0)
    fp = (pred_mask == 1) & (true_mask == 0)
    fn = (pred_mask == 0) & (true_mask == 1)
    
    confusion_map[tp] = [41, 175, 52]   # Green for TP
    confusion_map[tn] = [255, 255, 255] # White for TN
    confusion_map[fp] = [199, 0, 57]    # Red for FP
    confusion_map[fn] = [0, 0, 255]     # Blue for FN
    return confusion_map

def plot_predictions(conf_maps, model_names, folder_name, save_path):
    """Plot confusion maps for all models"""
    display_titles = {
        'U_Net': 'U-Net',
        'AttU_Net': 'Attention U-Net',
        'AttentionResUNet': 'Attention ResU-Net',
        'NestedUNet_He': 'UNet++',
        'NestedUnet_DeepSup': 'UNet++ DS',
        'UNet_3Plus_DeepSup': 'UNet 3+'
    }
    
    num_models = len(conf_maps)
    plt.figure(figsize=(5 * num_models, 5))
    
    for i, (conf_map, model_name) in enumerate(zip(conf_maps, model_names)):
        plt.subplot(1, num_models, i + 1)
        plt.imshow(conf_map)
        # title = display_titles.get(model_name, model_name)
        # plt.title(title, fontsize=12)
        plt.axis('off')
    
    # plt.gcf().text(0.02, 0.5, folder_name, fontsize=14, rotation=90, va='center', ha='right')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def has_two_numbers(filename):
    pattern = r'\d+_\d+\.npy$'
    return re.search(pattern, filename) is not None

def prepare_data(mod_type, size):
    """Prepare and filter data paths"""
    # Paths to images and masks
    imgs_patch = f'/share/wildfire-3/Sohel/Model/Data/Data/Main_2/{mod_type}/Patch_256_4/Data/'
    msks_patch = f'/share/wildfire-3/Sohel/Model/Data/Data/Main_2/{mod_type}/Patch_256_4/Mask/' 

    image_paths_patch = sorted(list(iglob(join(imgs_patch, '*.npy'), recursive=True)))
    mask_paths_patch = sorted(list(iglob(join(msks_patch, '*.npy'), recursive=True)))

    # Filter out specific filenames
    filtered_image_paths = [img for img in image_paths_patch if not has_two_numbers(os.path.basename(img))]
    filtered_mask_paths = [mask for mask in mask_paths_patch if not has_two_numbers(os.path.basename(mask))]

    # Train/test split
    train_filenames = ['laTorre','Avila','Pedrogao','BaraodeSaoJoao','Marinha',
    'Folgoso','AncientOlympia','ValldEbo', 'Ermida','Drosopigi','Makrimalli',
    'Sierra','Patras','Bejis']
    
    test_filenames = ['Patras','BaraodeSaoJoao',
    'AncientOlympia','Jubrique','VilaDeRei','Sierra',
    'Salakos','VilarinhodeSamarda',
    'Kalamos']

    train_images_filtered = [img for img in filtered_image_paths if os.path.basename(img).split('_')[0] in train_filenames]
    test_images_filtered = [img for img in filtered_image_paths if os.path.basename(img).split('_')[0] in test_filenames]
    train_masks_filtered = [mask for mask in filtered_mask_paths if os.path.basename(mask).split('_')[0] in train_filenames]
    test_masks_filtered = [mask for mask in filtered_mask_paths if os.path.basename(mask).split('_')[0] in test_filenames]

    return train_images_filtered, test_images_filtered, train_masks_filtered, test_masks_filtered

import rasterio
from rasterio.transform import from_origin

def save_confusion_geotiff(pred_mask, true_mask, dnbr_path, save_path):
    """Save confusion map (0: TN, 1: TP, 2: FP, 3: FN) as GeoTIFF using dNBR raster georeference."""
    with rasterio.open(dnbr_path) as src:
        dnbr_data = src.read(1)
        dnbr_meta = src.meta.copy()
        dnbr_transform = src.transform
        dnbr_crs = src.crs

        dnbr_height, dnbr_width = dnbr_data.shape

    # Match shape
    height = min(pred_mask.shape[0], true_mask.shape[0], dnbr_height)
    width = min(pred_mask.shape[1], true_mask.shape[1], dnbr_width)

    # Crop masks
    pred_mask = pred_mask[:height, :width]
    true_mask = true_mask[:height, :width]

    # Confusion map
    confusion_array = np.full((height, width), fill_value=255, dtype=np.uint8)
    tp = (pred_mask == 1) & (true_mask == 1)
    tn = (pred_mask == 0) & (true_mask == 0)
    fp = (pred_mask == 1) & (true_mask == 0)
    fn = (pred_mask == 0) & (true_mask == 1)

    confusion_array[tn] = 0
    confusion_array[tp] = 1
    confusion_array[fp] = 2
    confusion_array[fn] = 3

    # Update metadata
    dnbr_meta.update({
        "driver": "GTiff",
        "count": 1,
        "dtype": 'uint8',
        "height": height,
        "width": width,
        "transform": dnbr_transform,
        "crs": dnbr_crs,
        "compress": "lzw",
        "nodata": 255
    })

    with rasterio.open(save_path, "w", **dnbr_meta) as dst:
        dst.write(confusion_array, 1)

def main():
    set_random_seeds()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    data_types = ['Ind']  #[ 'Ind','Log','Combo','RF_Log','Log_2'] "cuda:1" if torch.cuda.is_available() else
    base_path = '/share/wildfire-3/Sohel/Model/Data/Data'
    channels_map = {
    'Log': 3,
    'Ind': 2,
    'Combo': 5,
    'RFDI': 1,
    'Log_2': 2
    }
    for data_type in data_types:
        # Set channels based on data type
        chnl = channels_map.get(data_type, 1)
        print(f"Processing {data_type} with {chnl} channels")
        
        
        # Define data paths
        data_paths = {
            "Log": {
                "input": f'/share/wildfire-3/Sohel/Model/Data/Data/{m}/Log/Data_2/',
                "mask": f'/share/wildfire-3/Sohel/Model/Data/Data/{m}/Log/Mask/'
            },
            "Ind": {
                "input": f'/share/wildfire-3/Sohel/Model/Data/Data/{m}/Ind/Data_2/',
                "mask": f'/share/wildfire-3/Sohel/Model/Data/Data/{m}/Ind/Mask/'
            },
            "Combo": {
                "input": f'/share/wildfire-3/Sohel/Model/Data/Data/{m}/Combo/Data_2/',
                "mask": f'/share/wildfire-3/Sohel/Model/Data/Data/{m}/Combo/Mask/'
            },
            "RFDI": {
                "input": f'/share/wildfire-3/Sohel/Model/Data/Data/{m}/RFDI/Data_2/',
                "mask": f'/share/wildfire-3/Sohel/Model/Data/Data/{m}/RFDI/Mask/'
            },
            "Log_2": {
                "input": f'/share/wildfire-3/Sohel/Model/Data/Data/{m}/Log_2/Data_2/',
                "mask": f'/share/wildfire-3/Sohel/Model/Data/Data/{m}/Log_2/Mask/'
            }
        }
        
       
        # Define test folders
        train_folders = []
        test_folders = ['Marinha','SeverDoVouga','Cugliericity',
        'GNDH','GSSC','LosGuajares','EastMani','Diabolitsi','Vatera']

        others = ['AncientOlympia','ValldEbo','VilarinhodeSamarda','AnonDeMoncayo','VilaDeRei',
        'Pinofranquendo','Salakos','Jubrique','Kechries','Freixianda','Ladrillar',
        'Monchique','LaDrova',
        'Prodromos','Mira','Kalamos','laTorre_1','Avila','Pedrogao',
        'BaraodeSaoJoao','Evia',
        'Marinha','Folgoso', 'Moros','Ermida','Drosopigi','Makrimalli',
        'Sierra','Patras','Bejis','Geraneia','CastroMarim']

        others = ['Patras','BaraodeSaoJoao',
    'AncientOlympia','Jubrique','VilaDeRei','Sierra',
    'Salakos','VilarinhodeSamarda',
    'Kalamos']

        # Get filtered and split data paths
        train_images, test_images, train_masks, test_masks = prepare_data(data_type, size=256)
        print(f"Found {len(train_images)} training images and {len(test_images)} test images")
        
        # Calculate normalization statistics using only training data
        # mean, std = calculate_normalization_stats(train_images, train_masks, chnl)
        # print(f"Calculated mean: {mean}")
        # print(f"Calculated std: {std}")
        
        input_path = data_paths[data_type]["input"]
        mask_path = data_paths[data_type]["mask"]
        
        image_files = [os.path.join(input_path, f'{folder}.npy') for folder in others]
        mask_files = [os.path.join(mask_path, f'{folder}.npy') for folder in others]

        dataset = UNetDataset_1(image_files, mask_files) #, mean, std
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

        for folder, (images, masks) in zip(others, dataloader):
            print(f"Processing folder: {folder}")            
            images, masks = images.to(device), masks.to(device)

            # Pad images and masks to be divisible by 32
            images = torch.stack([crop_to_divisible(img) for img in images])
            masks = torch.stack([crop_to_divisible(msk) for msk in masks])

            # Split images into patches
            patches = split_into_patches(images[0], patch_size=256)
            mask_patches = split_into_patches(masks[0], patch_size=256)

            model_results = []
            model_names = []

            dnbr_path = f'/share/wildfire-3/Sohel/Model/dNBR/dNBR_N/{folder}_dNBR_reclassified.tif'

            for model_name, model_constructor in model_architectures:
                print(f"Processing model: {model_name} for dataset {data_type}")
                model = model_constructor(chnl).to(device)
                model_weights_path = f'/share/wildfire-3/Sohel/Model/Data/Checkpoints/Models_30_1_Trial22/WithZeroMask/BCE_Final_NoAug_blc/MinMax_1/Trial-x-2/{data_type}_Final_256_0.001_{model_name}_Focal.pth'
                model.load_state_dict(torch.load(model_weights_path, map_location=device))
                model.eval()

                patch_predictions = []

                with torch.no_grad():
                    for (i, j, patch) in patches:
                        patch_tensor = patch.unsqueeze(0).to(device)
                        prediction = model(patch_tensor)
                        # Handle tuple predictions
                        if isinstance(prediction, tuple):
                            prediction = prediction[0]

                        prediction = prediction.squeeze().cpu().numpy()
                        binary_mask = (prediction > 0.5).astype(np.uint8)
                        patch_predictions.append((i, j, binary_mask))

                # Merge patches into a single prediction map
                merged_prediction = merge_patches(patch_predictions, images.shape, patch_size=256)

                # Adjust ground truth mask shape to match the merged prediction
                merged_prediction, ground_truth_mask = pad_to_match_shape(merged_prediction, masks[0, 0].cpu().numpy())

                # Generate confusion map
                conf_map = generate_confusion_map(merged_prediction, ground_truth_mask)

                model_results.append(conf_map)
                model_names.append(model_name)

            # Plot all models together for the current folder
            save_dir = f"/share/wildfire-3/Sohel/Model/Data/Figures/Models_30_1_Trial22/BCE_Final_NoAug_blc/geo/{data_type}/"
            os.makedirs(save_dir, exist_ok=True)

            # Save with folder name in filename and title
            # save_path = os.path.join(save_dir, f'{folder}.png')  # Save with the folder name
            # plot_predictions(model_results, model_names, folder, save_path)

            
            # Path to save georeferenced prediction
            geo_save_path = os.path.join(save_dir, f'{folder}.tif')

            # Save georeferenced prediction
            save_confusion_geotiff(merged_prediction, ground_truth_mask, dnbr_path, geo_save_path)

if __name__ == "__main__":
    main()
