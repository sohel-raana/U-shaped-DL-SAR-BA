"""
Sentinel-1 Preprocessing Module
Usage: 
    from preprocessing import Sentinel1Preprocessor
    preprocessor = Sentinel1Preprocessor()
    preprocessor.process_fire_case('Marinha')
"""

import os
import sys
import numpy as np
import pandas as pd
from glob import iglob
from os.path import join
from datetime import datetime, timedelta

# Add SNAP path
sys.path.append('/home/sohel/.snap/snap-python')
import snappy
from snappy import jpy, GPF, ProductIO


class Sentinel1Preprocessor:
    def __init__(self, base_path='/share/wildfire-3/Sohel'):
        """
        Initialize the Sentinel-1 preprocessor
        
        Args:
            base_path (str): Base path for data and outputs
        """
        self.base_path = base_path
        self.fire_data_path = f'{base_path}/Fire_Data.csv'
        
    def read_fire_data(self, fire_case):
        """
        Read fire data from CSV and get parameters for specific fire case
        
        Args:
            fire_case (str): Name of the fire case
            
        Returns:
            pandas.Series: Row containing fire case data
        """
        fire_data = pd.read_csv(self.fire_data_path)
        row = fire_data[fire_data['Fire_Case'] == fire_case]
        
        if row.empty:
            raise ValueError(f"Fire case '{fire_case}' not found in Fire_Data.csv")
        
        return row.iloc[0]
    
    def load_s1_data(self, product_path):
        """
        Load Sentinel-1 data files
        
        Args:
            product_path (str): Path containing S1 zip files
            
        Returns:
            list: List of loaded S1 products
        """
        input_S1_files = sorted(list(iglob(join(product_path, '*S1*.zip'), recursive=True)))
        
        if not input_S1_files:
            raise ValueError(f"No Sentinel-1 files found in {product_path}")
        
        print(f"Found {len(input_S1_files)} Sentinel-1 files")
        
        s1_data = []
        for i in input_S1_files:
            try:
                s1_read = snappy.ProductIO.readProduct(i)
                if s1_read is None:
                    print(f"Warning: Could not read file {i}. Skipping.")
                    continue
                
                name = s1_read.getName()
                print(f'Reading {name}')
                s1_data.append(s1_read)
            
            except Exception as e:
                print(f"An error occurred while processing file {i}: {e}")
        
        if not s1_data:
            raise ValueError("No valid Sentinel-1 data could be read")
        
        return s1_data
    
    def apply_orbit_file(self, product):
        """Apply orbit file correction"""
        parameters = snappy.HashMap()
        parameters.put('Apply-Orbit-File', True)
        return snappy.GPF.createProduct("Apply-Orbit-File", parameters, product)

    def thermal_noise_removal(self, product):
        """Remove thermal noise"""
        parameters = snappy.HashMap()
        parameters.put('removeThermalNoise', True)
        return snappy.GPF.createProduct('thermalNoiseRemoval', parameters, product)

    def calibration(self, product):
        """Perform radiometric calibration"""
        parameters = snappy.HashMap()
        parameters.put('selectedPolarisations', 'VH,VV')
        parameters.put('outputBetaBand', True)
        parameters.put('outputSigmaBand', False)
        parameters.put('outputImageScaleInDb', False)
        return snappy.GPF.createProduct('Calibration', parameters, product)

    def terrain_flattening(self, product):
        """Apply terrain flattening"""
        parameters = snappy.HashMap()
        return snappy.GPF.createProduct('Terrain-Flattening', parameters, product)

    def terrain_correction(self, product, proj):
        """Apply terrain correction with specified projection"""
        parameters = snappy.HashMap()
        parameters.put('demName', 'SRTM 1Sec HGT')
        parameters.put('pixelSpacingInMeter', 10.0)
        parameters.put('mapProjection', proj)
        parameters.put('nodataValueAtSea', False)
        return snappy.GPF.createProduct('Terrain-Correction', parameters, product)

    def subset(self, product, geom):
        """Create subset using geometry"""
        parameters = snappy.HashMap()
        parameters.put('geoRegion', geom)
        parameters.put('copyMetadata', True)
        return snappy.GPF.createProduct('Subset', parameters, product)

    def create_stack(self, product_list):
        """Create stack from multiple products"""
        parameters = snappy.HashMap()
        parameters.put('extent', 'Master')
        parameters.put('initialOffsetMethod', 'Product Geolocation')
        return snappy.GPF.createProduct("CreateStack", parameters, product_list)

    def multitemporal_filter(self, product):
        """Apply multitemporal speckle filter"""
        parameters = snappy.HashMap()
        parameters.put('filter', 'Lee')
        parameters.put('filterSizeX', 5)
        parameters.put('filterSizeY', 5)
        return snappy.GPF.createProduct('Multi-Temporal-Speckle-Filter', parameters, product)

    def sar_preprocessing_workflow(self, collection, proj, geom):
        """
        Execute the complete SAR preprocessing workflow
        
        Args:
            collection (list): List of S1 products
            proj (str): Projection string
            geom: Geometry object for subsetting
            
        Returns:
            list: List of preprocessed products
        """
        pre_list = []
        for i in collection:
            n = i.getName()
            print(f"Processing {n}")
            
            if '_Cal' in n:
                # Skip some steps for already calibrated data
                d = self.terrain_flattening(i)
                e = self.terrain_correction(d, proj)
                f = self.subset(e, geom)
                pre_list.append(f)
            else:
                a = self.apply_orbit_file(i)
                b = self.thermal_noise_removal(a)
                c = self.calibration(b)
                d = self.terrain_flattening(c)
                e = self.terrain_correction(d, proj)
                f = self.subset(e, geom)
                pre_list.append(f)
        
        return pre_list

    def get_ymd(self, txt):
        """
        Extract date from band name
        
        Args:
            txt (str): Band name containing date
            
        Returns:
            datetime: Parsed date
        """
        txt = txt.split('_')[-1]
        months = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
        }
        
        day = int(txt[:2])
        month_str = txt[2:5]
        month = months[month_str]
        year = int(txt[5:])
        
        return datetime(year, month, day)

    def separate_pre_post_fire_images(self, bands_name, fire_date):
        """
        Separate images into pre and post-fire based on fire date
        
        Args:
            bands_name (list): List of band names
            fire_date (datetime): Fire start date
            
        Returns:
            tuple: Lists of pre and post fire images for VH and VV
        """
        times = []
        for band in bands_name:
            times.append(self.get_ymd(band))

        sr1 = pd.Series(times, name='time')
        sr2 = pd.Series(bands_name, name='img')
        df = pd.concat((sr1, sr2), axis=1)
        df = df.sort_values('time')

        # Filter for VV and VH
        vv_files = df[df['img'].str.contains('VV')]
        vh_files = df[df['img'].str.contains('VH')]

        VH_imgs = vh_files['img'].tolist()
        VV_imgs = vv_files['img'].tolist()

        # Separate pre and post-fire images
        pre_fire_imgs_VH = [img for img in VH_imgs if self.get_ymd(img) < fire_date]
        post_fire_imgs_VH = [img for img in VH_imgs if self.get_ymd(img) >= fire_date]
        pre_fire_imgs_VV = [img for img in VV_imgs if self.get_ymd(img) < fire_date]
        post_fire_imgs_VV = [img for img in VV_imgs if self.get_ymd(img) >= fire_date]

        return pre_fire_imgs_VH, post_fire_imgs_VH, pre_fire_imgs_VV, post_fire_imgs_VV

    def create_time_average_bands(self, filtered_product, pre_fire_imgs_VH, post_fire_imgs_VH, 
                                 pre_fire_imgs_VV, post_fire_imgs_VV):
        """
        Create time-averaged bands for pre and post-fire periods
        
        Args:
            filtered_product: Filtered SAR product
            pre_fire_imgs_VH, post_fire_imgs_VH: Pre and post fire VH images
            pre_fire_imgs_VV, post_fire_imgs_VV: Pre and post fire VV images
            
        Returns:
            Product with time-averaged bands
        """
        # Generate expressions for time averaging
        exp_prevh = f"({' + '.join(pre_fire_imgs_VH)}) / {len(pre_fire_imgs_VH)}"
        exp_postvh = f"({' + '.join(post_fire_imgs_VH)}) / {len(post_fire_imgs_VH)}"
        exp_prevv = f"({' + '.join(pre_fire_imgs_VV)}) / {len(pre_fire_imgs_VV)}"
        exp_postvv = f"({' + '.join(post_fire_imgs_VV)}) / {len(post_fire_imgs_VV)}"

        # Create time-averaged bands
        BandDescriptor = jpy.get_type('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor')
        
        band_configs = [
            ('TimeAverage_VH_PreFire', exp_prevh),
            ('TimeAverage_VH_PostFire', exp_postvh),
            ('TimeAverage_VV_PreFire', exp_prevv),
            ('TimeAverage_VV_PostFire', exp_postvv)
        ]
        
        targetBands_avg = []
        for name, expr in band_configs:
            band = BandDescriptor()
            band.name = name
            band.type = 'float32'
            band.expression = expr
            targetBands_avg.append(band)

        targetBands = jpy.array('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor', len(targetBands_avg))
        for i, band in enumerate(targetBands_avg):
            targetBands[i] = band

        parameters = snappy.HashMap()
        parameters.put('targetBands', targetBands)
        
        return GPF.createProduct('BandMaths', parameters, filtered_product)

    def create_indices(self, time_average_product):
        """
        Create various SAR-based fire indices
        
        Args:
            time_average_product: Product with time-averaged bands
            
        Returns:
            Product with calculated indices
        """
        BandDescriptor = jpy.get_type('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor')
        
        # Define index expressions
        index_expressions = [
            ('RBD_VH', 'TimeAverage_VH_PostFire - TimeAverage_VH_PreFire'),
            ('RBD_VV', 'TimeAverage_VV_PostFire - TimeAverage_VV_PreFire'),
            ('LogRBR_VH', 'log10(TimeAverage_VH_PostFire / TimeAverage_VH_PreFire)'),
            ('LogRBR_VV', 'log10(TimeAverage_VV_PostFire / TimeAverage_VV_PreFire)'),
            ('DeltaRVI', '4 * TimeAverage_VH_PostFire/(TimeAverage_VV_PostFire + TimeAverage_VH_PostFire) - 4 * TimeAverage_VH_PreFire/(TimeAverage_VV_PreFire + TimeAverage_VH_PreFire)'),
            ('DeltaDPSVI', '(TimeAverage_VV_PostFire + TimeAverage_VH_PostFire)/TimeAverage_VV_PostFire - (TimeAverage_VV_PreFire + TimeAverage_VH_PreFire)/TimeAverage_VV_PreFire'),
            ('RFDI', '(TimeAverage_VV_PostFire - TimeAverage_VH_PostFire)/(TimeAverage_VV_PostFire + TimeAverage_VH_PostFire) - (TimeAverage_VV_PreFire - TimeAverage_VH_PreFire)/(TimeAverage_VV_PreFire + TimeAverage_VH_PreFire)'),
            ('LogRBR_CR', 'log10((TimeAverage_VV_PostFire / TimeAverage_VH_PostFire)/(TimeAverage_VV_PreFire / TimeAverage_VH_PreFire))')
        ]

        targetBands_indices = []
        for name, expr in index_expressions:
            band = BandDescriptor()
            band.name = name
            band.type = 'float32'
            band.expression = expr
            targetBands_indices.append(band)

        targetBands = jpy.array('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor', len(targetBands_indices))
        for i, band in enumerate(targetBands_indices):
            targetBands[i] = band

        parameters = snappy.HashMap()
        parameters.put('targetBands', targetBands)

        return GPF.createProduct('BandMaths', parameters, time_average_product)

    def process_fire_case(self, fire_case):
        """
        Process a complete fire case from raw S1 data to indices
        
        Args:
            fire_case (str): Name of the fire case to process
            
        Returns:
            str: Path to output indices file
        """
        print(f"Starting preprocessing for fire case: {fire_case}")
        
        # Get fire case parameters
        fire_row = self.read_fire_data(fire_case)
        product_path = f'{self.base_path}/Data/{fire_case}'
        
        # Load S1 data
        s1_data = self.load_s1_data(product_path)
        
        # Get projection and geometry
        proj = fire_row['Proj']
        wkt = fire_row['WKT']
        geom = snappy.WKTReader().read(wkt)
        
        # Run preprocessing workflow
        print("Running SAR preprocessing workflow...")
        processed_list = self.sar_preprocessing_workflow(s1_data, proj, geom)
        
        # Create stack and apply filter
        stack = self.create_stack(processed_list)
        filtered = self.multitemporal_filter(stack)
        
        # Get band names and process dates
        bands = filtered.getBandNames()
        bands_name = list(bands)
        
        # Parse fire date
        fire_strt = fire_row['Fire_Start_Date']
        year, month, day = map(int, fire_strt.split(','))
        fire_date = datetime(year, month, day)
        
        # Separate pre and post-fire images
        pre_fire_imgs_VH, post_fire_imgs_VH, pre_fire_imgs_VV, post_fire_imgs_VV = \
            self.separate_pre_post_fire_images(bands_name, fire_date)
        
        print(f"Pre-fire VH images: {len(pre_fire_imgs_VH)}")
        print(f"Post-fire VH images: {len(post_fire_imgs_VH)}")
        print(f"Pre-fire VV images: {len(pre_fire_imgs_VV)}")
        print(f"Post-fire VV images: {len(post_fire_imgs_VV)}")
        
        # Create time-averaged bands
        print("Creating time-averaged bands...")
        time_average = self.create_time_average_bands(
            filtered, pre_fire_imgs_VH, post_fire_imgs_VH, 
            pre_fire_imgs_VV, post_fire_imgs_VV
        )
        
        # Create indices
        print("Creating fire indices...")
        indices = self.create_indices(time_average)
        
        # Write output
        output_path = f'{self.base_path}/Output/Indices/{fire_case}'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f'Writing indices to: {output_path}')
        ProductIO.writeProduct(indices, output_path, "BEAM-DIMAP")
        
        print(f"Preprocessing completed for {fire_case}")
        return output_path


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sentinel-1 Preprocessing')
    parser.add_argument('--case', required=True, help='Fire case name')
    parser.add_argument('--base_path', default='/share/wildfire-3/Sohel', help='Base path for data')
    
    args = parser.parse_args()
    
    preprocessor = Sentinel1Preprocessor(args.base_path)
    output_path = preprocessor.process_fire_case(args.case)
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()