# Data Structure
The pipeline expects the following directory structure:

```
/share/wildfire-3/Sohel/  
├── Data/  
│   └── {FireCase}/               # Raw Sentinel-1 zip files  
├── Fire_Data.csv                 # Fire metadata (dates, projections, WKT)  
├── Model/  
│   ├── Data/  
│   │   ├── Data/Main_2/{DataType}/  
│   │   │   ├── Data_2/           # Processed input data (.npy)  
│   │   │   └── Mask/             # Ground truth masks (.npy)  
│   │   └── Checkpoints/          # Trained model weights  
│   └── dNBR/                     # dNBR reference rasters  
└── Output/  
    └── Indices/                  # Preprocessed SAR indices
```

Usage  
Command Line Interface  
```
python predict.py --case <fire_case_name> [OPTIONS]  
```
Basic Usage Examples  
Run complete pipeline (preprocessing + prediction):  
```
python predict.py --case Marinha --data_type Ind  
```
Skip preprocessing (use existing processed data):  
```
python predict.py --case Marinha --data_type Ind --skip_preprocessing  
```
Use different data type:  
```
python predict.py --case Marinha --data_type Log  
```
Custom base path:  
```
python predict.py --case Marinha --data_type Ind --base_path /custom/path  
```  
Outputs  
For each processed fire case, the pipeline generates:  
```
Results/{FireCase}/{DataType}/
├── {FireCase}_{ModelName}_results.png      # Confusion map + metrics plot
├── {FireCase}_{ModelName}_confusion.tif    # Georeferenced confusion map
└── {FireCase}_{ModelName}_metrics.csv      # Detailed performance metrics
```
