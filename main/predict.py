"""
Automated Fire Prediction Pipeline
Integrates preprocessing, prediction, and evaluation in a single command.

Usage:
    python predict.py --case Marinha --data_type Ind --base_path /share/wildfire-3/Sohel
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import rasterio
from rasterio.transform import from_origin

# Import preprocessing module
from preprocessing import Sentinel1Preprocessor

# Add model paths
sys.path.append('/share/wildfire-3/Sohel/Model/Code/UNet')

# Import model modules
from DataLoad import UNetDataset_1
from ExtendedNestedUNet import NestedUNetHeTrans, NestedUnet_DeepSup
from UNet3Plus import UNet_3Plus_DeepSup


class FirePredictionPipeline:
    def __init__(self, base_path='/share/wildfire-3/Sohel', gpu=0):
        """
        Initialize the fire prediction pipeline
        
        Args:
            base_path (str): Base path for data and outputs
        """
        self.base_path = base_path
        self.preprocessor = Sentinel1Preprocessor(base_path)
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu}")
        else:
            self.device = torch.device("cpu")
        
        # Model architectures
        self.model_architectures = [
            ('NestedUnet_DeepSup', lambda c: NestedUnet_DeepSup(c, 1)),
        ]
        
        # Data type configurations
        self.channels_map = {
            'Log': 3,
            'Ind': 2,
            'Combo': 5,
            'RFDI': 1,
            'Log_2': 2
        }
        
        self.data_paths = {
            "Log": {
                "input": f'{base_path}/Model/Data/Data/Main_2/Log/Data_2/',
                "mask": f'{base_path}/Model/Data/Data/Main_2/Log/Mask/'
            },
            "Ind": {
                "input": f'{base_path}/Model/Data/Data/Main_2/Ind/Data_2/',
                "mask": f'{base_path}/Model/Data/Data/Main_2/Ind/Mask/'
            },
            "Combo": {
                "input": f'{base_path}/Model/Data/Data/Main_2/Combo/Data_2/',
                "mask": f'{base_path}/Model/Data/Data/Main_2/Combo/Mask/'
            },
            "RFDI": {
                "input": f'{base_path}/Model/Data/Data/Main_2/RFDI/Data_2/',
                "mask": f'{base_path}/Model/Data/Data/Main_2/RFDI/Mask/'
            },
            "Log_2": {
                "input": f'{base_path}/Model/Data/Data/Main_2/Log_2/Data_2/',
                "mask": f'{base_path}/Model/Data/Data/Main_2/Log_2/Mask/'
            }
        }
    
    def set_random_seeds(self, seed=42):
        """Set random seeds for reproducibility"""
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    def pad_to_match_shape(self, arr1, arr2):
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
    
    def crop_to_divisible(self, image, divisor=32):
        """Crop image so that its dimensions are divisible by a given divisor"""
        h, w = image.shape[-2:]
        new_h = h - (h % divisor)
        new_w = w - (w % divisor)
        return image[..., :new_h, :new_w]
    
    def split_into_patches(self, image, patch_size=256):
        """Split image into patches"""
        h, w = image.shape[-2:]
        patches = []
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                patch = image[:, i:i + patch_size, j:j + patch_size]
                patches.append((i, j, patch))
        return patches
    
    def merge_patches(self, patch_predictions, image_shape, patch_size=256):
        """Merge predictions back into full image"""
        h, w = image_shape[-2:]
        merged = np.zeros((h, w), dtype=np.float32)
        for (i, j, patch_pred) in patch_predictions:
            patch_h, patch_w = patch_pred.shape
            merged[i:i + patch_h, j:j + patch_w] = patch_pred
        return merged
    
    def generate_confusion_map(self, pred_mask, true_mask):
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
    
    def calculate_metrics(self, pred_mask, true_mask):
        """Calculate evaluation metrics"""
        pred_flat = pred_mask.flatten()
        true_flat = true_mask.flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(true_flat, pred_flat)
        precision = precision_score(true_flat, pred_flat, zero_division=0)
        recall = recall_score(true_flat, pred_flat, zero_division=0)
        f1 = f1_score(true_flat, pred_flat, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(true_flat, pred_flat)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }
        
        return metrics
    
    def save_confusion_geotiff(self, pred_mask, true_mask, dnbr_path, save_path):
        """Save confusion map as GeoTIFF using dNBR raster georeference."""
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
    
    def plot_results(self, confusion_map, metrics, case_name, save_path):
        """Plot confusion map with metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot confusion map
        ax1.imshow(confusion_map)
        ax1.set_title(f'Fire Prediction Results - {case_name}', fontsize=14)
        ax1.axis('off')
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=np.array([41, 175, 52])/255, label='True Positive'),
            plt.Rectangle((0, 0), 1, 1, facecolor=np.array([255, 255, 255])/255, label='True Negative'),
            plt.Rectangle((0, 0), 1, 1, facecolor=np.array([199, 0, 57])/255, label='False Positive'),
            plt.Rectangle((0, 0), 1, 1, facecolor=np.array([0, 0, 255])/255, label='False Negative')
        ]
        ax1.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        # Plot metrics
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score']]
        
        bars = ax2.bar(metric_names, metric_values, color=['#2E8B57', '#4169E1', '#FF6347', '#FFD700'])
        ax2.set_title('Performance Metrics', fontsize=14)
        ax2.set_ylabel('Score')
        ax2.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def run_preprocessing(self, fire_case):
        """Run preprocessing for the fire case"""
        print(f"Starting preprocessing for {fire_case}...")
        try:
            output_path = self.preprocessor.process_fire_case(fire_case)
            print(f"Preprocessing completed successfully: {output_path}")
            return True
        except Exception as e:
            print(f"Preprocessing failed: {str(e)}")
            return False
    
    def run_prediction(self, fire_case, data_type='Ind'):
        """Run prediction and evaluation for the fire case"""
        print(f"Starting prediction for {fire_case} with data type {data_type}...")
        
        self.set_random_seeds()
        
        # Get channels for data type
        chnl = self.channels_map.get(data_type, 1)
        
        # Prepare data paths
        input_path = self.data_paths[data_type]["input"]
        mask_path = self.data_paths[data_type]["mask"]
        
        image_file = os.path.join(input_path, f'{fire_case}.npy')
        mask_file = os.path.join(mask_path, f'{fire_case}.npy')
        
        # Check if files exist
        if not os.path.exists(image_file):
            print(f"Error: Image file not found: {image_file}")
            return None
        if not os.path.exists(mask_file):
            print(f"Error: Mask file not found: {mask_file}")
            return None
        
        # Load data
        print("Loading data...")
        dataset = UNetDataset_1([image_file], [mask_file])
        images, masks = dataset[0]
        
        images = images.unsqueeze(0).to(self.device)
        masks = masks.unsqueeze(0).to(self.device)
        
        # Crop images and masks to be divisible by 32
        images = torch.stack([self.crop_to_divisible(img) for img in images])
        masks = torch.stack([self.crop_to_divisible(msk) for msk in masks])
        
        # Split images into patches
        patches = self.split_into_patches(images[0], patch_size=256)
        
        results = {}
        
        for model_name, model_constructor in self.model_architectures:
            print(f"Processing model: {model_name}")
            
            # Load model
            model = model_constructor(chnl).to(self.device)
            model_weights_path = f'{self.base_path}/Model/Data/Checkpoints/Models_30_1_Trial22/WithZeroMask/BCE_Final_NoAug_blc/MinMax_1/Trial-x-2/{data_type}_Final_256_0.001_{model_name}_Focal.pth'
            
            if not os.path.exists(model_weights_path):
                print(f"Warning: Model weights not found: {model_weights_path}")
                continue
                
            model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
            model.eval()
            
            patch_predictions = []
            
            with torch.no_grad():
                for (i, j, patch) in patches:
                    patch_tensor = patch.unsqueeze(0).to(self.device)
                    prediction = model(patch_tensor)
                    
                    # Handle tuple predictions
                    if isinstance(prediction, tuple):
                        prediction = prediction[0]
                    
                    prediction = prediction.squeeze().cpu().numpy()
                    binary_mask = (prediction > 0.5).astype(np.uint8)
                    patch_predictions.append((i, j, binary_mask))
            
            # Merge patches
            merged_prediction = self.merge_patches(patch_predictions, images.shape, patch_size=256)
            
            # Adjust ground truth mask shape
            merged_prediction, ground_truth_mask = self.pad_to_match_shape(
                merged_prediction, masks[0, 0].cpu().numpy()
            )
            
            # Generate confusion map
            conf_map = self.generate_confusion_map(merged_prediction, ground_truth_mask)
            
            # Calculate metrics
            metrics = self.calculate_metrics(merged_prediction, ground_truth_mask)
            
            results[model_name] = {
                'prediction': merged_prediction,
                'ground_truth': ground_truth_mask,
                'confusion_map': conf_map,
                'metrics': metrics
            }
        
        return results
    
    def save_results(self, results, fire_case, data_type='Ind'):
        """Save all results including plots, metrics, and GeoTIFFs"""
        save_dir = f"{self.base_path}/Model/Data/Results/{fire_case}/{data_type}/"
        os.makedirs(save_dir, exist_ok=True)
        
        # dNBR path for georeferencing
        dnbr_path = f'{self.base_path}/Model/dNBR/dNBR_N/{fire_case}_dNBR_reclassified.tif'
        
        for model_name, result in results.items():
            # Save plot with metrics
            plot_save_path = os.path.join(save_dir, f'{fire_case}_{model_name}_results.png')
            self.plot_results(
                result['confusion_map'], 
                result['metrics'], 
                f"{fire_case} - {model_name}", 
                plot_save_path
            )
            
            # Save GeoTIFF if dNBR exists
            if os.path.exists(dnbr_path):
                geotiff_save_path = os.path.join(save_dir, f'{fire_case}_{model_name}_confusion.tif')
                self.save_confusion_geotiff(
                    result['prediction'], 
                    result['ground_truth'], 
                    dnbr_path, 
                    geotiff_save_path
                )
                print(f"Saved GeoTIFF: {geotiff_save_path}")
            
            # Save metrics to CSV
            metrics_df = pd.DataFrame([result['metrics']])
            metrics_df['fire_case'] = fire_case
            metrics_df['model'] = model_name
            metrics_df['data_type'] = data_type
            metrics_df['timestamp'] = datetime.now().isoformat()
            
            metrics_save_path = os.path.join(save_dir, f'{fire_case}_{model_name}_metrics.csv')
            metrics_df.to_csv(metrics_save_path, index=False)
            
            # Print metrics
            print(f"\nMetrics for {model_name}:")
            print(f"Accuracy: {result['metrics']['accuracy']:.4f}")
            print(f"Precision: {result['metrics']['precision']:.4f}")
            print(f"Recall: {result['metrics']['recall']:.4f}")
            print(f"F1 Score: {result['metrics']['f1_score']:.4f}")
            print(f"Results saved to: {save_dir}")
    
    def run_full_pipeline(self, fire_case, data_type='Ind', run_preprocessing=True):
        """Run the complete pipeline: preprocessing -> prediction -> evaluation"""
        print(f"Starting full pipeline for {fire_case}")
        print("="*50)
        
        # Step 1: Preprocessing (optional)
        if run_preprocessing:
            print("Step 1: Running preprocessing...")
            if not self.run_preprocessing(fire_case):
                print("Pipeline stopped due to preprocessing failure.")
                return None
        else:
            print("Step 1: Skipping preprocessing (using existing data)")
        
        # Step 2: Prediction and evaluation
        print("\nStep 2: Running prediction and evaluation...")
        results = self.run_prediction(fire_case, data_type)
        
        if results is None:
            print("Pipeline stopped due to prediction failure.")
            return None
        
        # Step 3: Save results
        print("\nStep 3: Saving results...")
        self.save_results(results, fire_case, data_type)
        
        print("\n" + "="*50)
        print("Pipeline completed successfully!")
        
        return results


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Fire Prediction Pipeline')
    parser.add_argument('--case', required=True, help='Fire case name (e.g., Marinha)')
    parser.add_argument('--data_type', default='Ind', choices=['Log', 'Ind', 'Combo', 'RFDI', 'Log_2'],
                       help='Data type for prediction')
    parser.add_argument('--base_path', default='/share/wildfire-3/Sohel', help='Base path for data')
    parser.add_argument('--skip_preprocessing', action='store_true', 
                       help='Skip preprocessing step (use existing processed data)')
    parser.add_argument('--gpu', type=int, default=0,
                   help='GPU id to use (e.g., 0 or 1). Ignored if no CUDA available')
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = FirePredictionPipeline(args.base_path, gpu=args.gpu)
    results = pipeline.run_full_pipeline(
        fire_case=args.case,
        data_type=args.data_type,
        run_preprocessing=not args.skip_preprocessing
    )
    
    if results:
        print(f"\nPipeline completed successfully for {args.case}")
        print(f"Check results in: {args.base_path}/Model/Data/Results/{args.case}/{args.data_type}/")
    else:
        print(f"\nPipeline failed for {args.case}")


if __name__ == "__main__":
    main()
