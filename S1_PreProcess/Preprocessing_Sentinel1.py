'''PRE-PROCESSING'''

import sys
sys.path.append('/home/sohel/.snap/snap-python')
import snappy

from os.path import join
import subprocess
from glob import iglob
from zipfile import ZipFile
import numpy as np
import pandas as pd
import snappy
from snappy import jpy, GPF, ProductIO
import matplotlib.pyplot as plt

#set target folder and extract metadata
Fire_Cases = ['Creek_Fire']
# Fire_Cases = ["Makrimalli",
# 'Monchique', 'VilaDeRei', 'Kalamos','Marinha','Patras','Ermida',
# 'Pedrogao','laTorre_1', 'BaraodeSaoJoao', 
# 'Pinofranquendo', 'LaDrova', 'Mira','Salakos','Valleseco', 'Bejis','Geraneia',"AncientOlympia",'Drosopigi','Avila','Sierra','CastroMarim',
# 'Lithakia','Prodromos','VilarinhodeSamarda','Talhadas','Jubrique','Kechries']

for Fire_Case in Fire_Cases:
    Fire_data = pd.read_csv('/share/wildfire-3/Sohel/Fire_Data.csv')
    product_path = f'/share/wildfire-3/Sohel/Data/{Fire_Case}'
    input_S1_files = sorted(list(iglob(join(product_path, '*S1*.zip'), recursive=True)))
    row = Fire_data[Fire_data['Fire_Case'] == Fire_Case]

    s1_data = []  
    for i in input_S1_files:
        try:
            s1_read = snappy.ProductIO.readProduct(i)
            if s1_read is None:
                print(f"Warning: Could not read file {i}. Skipping.")
                continue  # Skip to the next file
            
            name = s1_read.getName()
            print(f'Reading {name}')
            s1_data.append(s1_read)
        
        except Exception as e:
            print(f"An error occurred while processing file {i}: {e}")

    def read(filename):
        return ProductIO.readProduct(filename)

    def write(product, filename):
        ProductIO.writeProduct(product, filename, "BEAM-DIMAP")

    '''APPLY ORBIT FILE'''
    def apply_orbit_file(product):
        parameters = snappy.HashMap()
        parameters.put('Apply-Orbit-File', True)    
        return snappy.GPF.createProduct("Apply-Orbit-File", parameters, product)

    '''THERMAL NOISE REMOVAL'''
    def thermal_noise_removal(product):
        parameters = snappy.HashMap()
        parameters.put('removeThermalNoise', True)
        return snappy.GPF.createProduct('thermalNoiseRemoval', parameters, product)

    '''CALIBRATION'''
    def calibration(product):
        parameters = snappy.HashMap()
        parameters.put('selectedPolarisations', 'VH,VV')
        parameters.put('outputBetaBand', True)
        parameters.put('outputSigmaBand', False)
        parameters.put('outputImageScaleInDb', False)
        return snappy.GPF.createProduct('Calibration', parameters, product)

    '''TERRAIN FLATTENING'''
    def terrain_flattening(product):
        parameters = snappy.HashMap()
        return snappy.GPF.createProduct('Terrain-Flattening', parameters, product)

    '''TERRAIN CORRECTION'''
    proj = row['Proj'].values[0]
    proj_PO = f'''{proj}''' 

    def terrain_correction(product, proj):
        parameters = snappy.HashMap()
        parameters.put('demName', 'SRTM 1Sec HGT')
        parameters.put('pixelSpacingInMeter', 10.0)
        parameters.put('mapProjection', proj)
        parameters.put('nodataValueAtSea', False)
        return snappy.GPF.createProduct('Terrain-Correction', parameters, product)

    '''SUBSET'''
    wkt_PO= row['WKT'].values[0]
    geom_PO = snappy.WKTReader().read(wkt_PO)

    def subset(product, geom):
        parameters = snappy.HashMap()
        parameters.put('geoRegion', geom)
        parameters.put('copyMetadata', True)
        return snappy.GPF.createProduct('Subset', parameters, product)

    '''COREGISTRATION: CREATE STACK'''
    def create_stack(product):
        parameters = snappy.HashMap()
        parameters.put('extent','Master')
        parameters.put('initialOffsetMethod', 'Product Geolocation')

        return snappy.GPF.createProduct("CreateStack", parameters, product)

    '''MULTITEMPORAL SPECKLE FILTER'''

    def multitemporal_filter(product):
        parameters = snappy.HashMap()
        parameters.put('filter', 'Lee')
        parameters.put('filterSizeX', 5)
        parameters.put('filterSizeY', 5)
        return snappy.GPF.createProduct('Multi-Temporal-Speckle-Filter', parameters, product)

    '''Process'''

    def SAR_preprocessing_workflow(collection):
        pre_list = []
        for i in collection:
            n = i.getName()
            print(n)
            if '_Cal' in n:
                # Perform subset step
                d = terrain_flattening(i)
                e = terrain_correction(d, proj_PO)
                f = subset(e, geom_PO)
                pre_list.append(f)
            else:
                a = apply_orbit_file(i)
                b = thermal_noise_removal(a)
                c = calibration(b)
                d = terrain_flattening(c)
                e = terrain_correction(d, proj_PO)
                f = subset(e, geom_PO)
                
                pre_list.append(f)
        
        return pre_list

    PL = SAR_preprocessing_workflow(s1_data)
    stack = create_stack(PL)
    filtered = multitemporal_filter(stack)
    #write(filtered, f'/share/wildfire-3/Sohel/Output/S1/NK2')

    bands = filtered.getBandNames()
    #print("Bands:%s" % (list(bands)))
    bands_name= list(bands)
    bands_name

    #---------------------------------------
    from datetime import datetime, timedelta
    getdate = lambda x: datetime.strptime(x.split('_')[-1], '%d%b%Y')

    def get_ymd(txt):
        txt  =  txt.split('_')[-1]
        # 월을 숫자로 매핑
        months = {
            'Jan': 1,
            'Feb': 2,
            'Mar': 3,
            'Apr': 4,
            'May': 5,
            'Jun': 6,
            'Jul': 7,
            'Aug': 8,
            'Sep': 9,
            'Oct': 10,
            'Nov': 11,
            'Dec': 12
        }
        
            # 문자열 분리
        day = int(txt[:2])
        month_str = txt[2:5]
        month = months[month_str]
        year = int(txt[5:])
        
        return datetime(year, month, day)

    times = []
    for i in range(len(bands_name)):
        times.append(get_ymd(bands_name[i]))

    sr1 = pd.Series(times, name = 'time')
    sr2 = pd.Series(bands_name, name = 'img')

    df = pd.concat((sr1, sr2), axis = 1)
    df = df.sort_values('time')
    list_len = int(len(df)/2)

    # Filter for VV and VH
    vv_files = df[df['img'].str.contains('VV')]
    vh_files = df[df['img'].str.contains('VH')]

    VH_imgs = vh_files['img'].tolist()
    VV_imgs = vv_files['img'].tolist()

    print(f"Image List: {VH_imgs}")

    # Define the fire date
    fire_strt = row['Fire_Start_Date'].values[0]
    year, month, day = map(int, fire_strt.split(','))
    fire_date = datetime(year, month, day)

    # Define the start and end dates for the 2-week pre-fire period
    start_date = fire_date - timedelta(weeks=1)
    end_date = fire_date

    # Separate pre and post-fire VH images
    pre_fire_imgs_VH = [img for img in VH_imgs if get_ymd(img)< fire_date]
    post_fire_imgs_VH = [img for img in VH_imgs if get_ymd(img) >= fire_date]
    pre_fire_imgs_VV = [img for img in VV_imgs if get_ymd(img)< fire_date]
    post_fire_imgs_VV = [img for img in VV_imgs if get_ymd(img) >= fire_date]

    # Filter pre-fire images to be within 2 weeks before the fire date
    # pre_fire_imgs_VH = [img for img in pre_fire_imgs_VH if start_date <= get_ymd(img) < end_date]
    # pre_fire_imgs_VV = [img for img in pre_fire_imgs_VV if start_date <= get_ymd(img) < end_date]

    # Generate expressions for pre and post-fire periods
    exp_prevh = f"({' + '.join(pre_fire_imgs_VH)}) / {len(pre_fire_imgs_VH)}"
    exp_postvh = f"({' + '.join(post_fire_imgs_VH)}) / {len(post_fire_imgs_VH)}"
    exp_prevv = f"({' + '.join(pre_fire_imgs_VV)}) / {len(pre_fire_imgs_VV)}"
    exp_postvv = f"({' + '.join(post_fire_imgs_VV)}) / {len(post_fire_imgs_VV)}"

    # Generate expressions for pre and post-fire periods
    exp_prevh_mean = f"({' + '.join(pre_fire_imgs_VH)}) / {len(pre_fire_imgs_VH)}"
    exp_postvh_mean = f"({' + '.join(post_fire_imgs_VH)}) / {len(post_fire_imgs_VH)}"
    exp_prevv_mean = f"({' + '.join(pre_fire_imgs_VV)}) / {len(pre_fire_imgs_VV)}"
    exp_postvv_mean = f"({' + '.join(post_fire_imgs_VV)}) / {len(post_fire_imgs_VV)}"

    # Expressions for standard deviation
    exp_prevh_std = f"sqrt(({' + '.join([f'sq({img} - {exp_prevh_mean})' for img in pre_fire_imgs_VH])}) / {len(pre_fire_imgs_VH)})"
    exp_postvh_std = f"sqrt(({' + '.join([f'sq({img} - {exp_postvh_mean})' for img in post_fire_imgs_VH])}) / {len(post_fire_imgs_VH)})"
    exp_prevv_std = f"sqrt(({' + '.join([f'sq({img} - {exp_prevv_mean})' for img in pre_fire_imgs_VV])}) / {len(pre_fire_imgs_VV)})"
    exp_postvv_std = f"sqrt(({' + '.join([f'sq({img} - {exp_postvv_mean})' for img in post_fire_imgs_VV])}) / {len(post_fire_imgs_VV)})"

    '''Creation of Time-Average'''
    BandDescriptor = jpy.get_type('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor')
    targetBand1 = BandDescriptor()
    targetBand1.name = 'TimeAverage_VH_PreFire'
    targetBand1.type = 'float32'
    targetBand1.expression = exp_prevh

    targetBand2 = BandDescriptor()
    targetBand2.name = 'TimeAverage_VH_PostFire'
    targetBand2.type = 'float32'
    targetBand2.expression = exp_postvh

    targetBand3 = BandDescriptor()
    targetBand3.name = 'TimeAverage_VV_PreFire'
    targetBand3.type = 'float32'
    targetBand3.expression = exp_prevv

    targetBand4 = BandDescriptor()
    targetBand4.name = 'TimeAverage_VV_PostFire'
    targetBand4.type = 'float32'
    targetBand4.expression = exp_postvv

    # Create BandDescriptors for standard deviation
    targetBand5 = BandDescriptor()
    targetBand5.name = 'StdDev_VH_PreFire'
    targetBand5.type = 'float32'
    targetBand5.expression = exp_prevh_std

    targetBand6 = BandDescriptor()
    targetBand6.name = 'StdDev_VH_PostFire'
    targetBand6.type = 'float32'
    targetBand6.expression = exp_postvh_std

    targetBand7 = BandDescriptor()
    targetBand7.name = 'StdDev_VV_PreFire'
    targetBand7.type = 'float32'
    targetBand7.expression = exp_prevv_std

    targetBand8 = BandDescriptor()
    targetBand8.name = 'StdDev_VV_PostFire'
    targetBand8.type = 'float32'
    targetBand8.expression = exp_postvv_std

    # Add new bands to targetBands
    targetBands = jpy.array('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor', 8)
    targetBands[0] = targetBand1
    targetBands[1] = targetBand2
    targetBands[2] = targetBand3
    targetBands[3] = targetBand4
    targetBands[4] = targetBand5
    targetBands[5] = targetBand6
    targetBands[6] = targetBand7
    targetBands[7] = targetBand8

    parameters = snappy.HashMap()
    parameters.put('targetBands', targetBands)

    TimeAverage = GPF.createProduct('BandMaths', parameters, filtered)

    bands = TimeAverage.getBandNames()
    bands_name= list(bands)

    '''Creation of Time-Average'''
    BandDescriptor = jpy.get_type('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor')
    targetBand1 = BandDescriptor()
    targetBand1.name = 'RBD_VH'
    targetBand1.type = 'float32'
    targetBand1.expression = 'TimeAverage_VH_PostFire - TimeAverage_VH_PreFire'

    targetBand2 = BandDescriptor()
    targetBand2.name = 'RBD_VV'
    targetBand2.type = 'float32'
    targetBand2.expression = 'TimeAverage_VV_PostFire - TimeAverage_VV_PreFire'

    targetBand3 = BandDescriptor()
    targetBand3.name = 'LogRBR_VH'
    targetBand3.type = 'float32'
    targetBand3.expression = 'log10(TimeAverage_VH_PostFire / TimeAverage_VH_PreFire)'

    targetBand4 = BandDescriptor()
    targetBand4.name = 'LogRBR_VV'
    targetBand4.type = 'float32'
    targetBand4.expression = 'log10(TimeAverage_VV_PostFire / TimeAverage_VV_PreFire)'

    RVI_post = '4 * TimeAverage_VH_PostFire/(TimeAverage_VV_PostFire + TimeAverage_VH_PostFire)'
    RVI_pre = '4 * TimeAverage_VH_PreFire/(TimeAverage_VV_PreFire + TimeAverage_VH_PreFire)'
    DPSVI_post = '(TimeAverage_VV_PostFire + TimeAverage_VH_PostFire)/TimeAverage_VV_PostFire'
    DPSVI_pre = '(TimeAverage_VV_PreFire + TimeAverage_VH_PreFire)/TimeAverage_VV_PreFire'

    targetBand5 = BandDescriptor()
    targetBand5.name = 'DeltaRVI'
    targetBand5.type = 'float32'
    targetBand5.expression = RVI_post + '-' + RVI_pre

    targetBand6 = BandDescriptor()
    targetBand6.name = 'DeltaDPSVI'
    targetBand6.type = 'float32'
    targetBand6.expression = DPSVI_post + '-' + DPSVI_pre

    RFDI_post = '(TimeAverage_VV_PostFire - TimeAverage_VH_PostFire)/(TimeAverage_VV_PostFire + TimeAverage_VH_PostFire)'
    RFDI_pre = '(TimeAverage_VV_PreFire - TimeAverage_VH_PreFire)/(TimeAverage_VV_PreFire + TimeAverage_VH_PreFire)'
    DeltaRFDI = RFDI_post + '-' + RFDI_pre

    BandDescriptor = jpy.get_type('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor')
    targetBand7 = BandDescriptor()
    targetBand7.name = 'RFDI'
    targetBand7.type = 'float32'
    targetBand7.expression = DeltaRFDI

    BandDescriptor = jpy.get_type('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor')
    targetBand8 = BandDescriptor()
    targetBand8.name = 'LogRBR_CR'
    targetBand8.type = 'float32'
    targetBand8.expression = 'log10((TimeAverage_VV_PostFire / TimeAverage_VH_PostFire)/(TimeAverage_VV_PreFire / TimeAverage_VH_PreFire))'

    targetBands = jpy.array('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor', 8)
    targetBands[0] = targetBand1
    targetBands[1] = targetBand2
    targetBands[2] = targetBand3
    targetBands[3] = targetBand4
    targetBands[4] = targetBand5
    targetBands[5] = targetBand6
    targetBands[6] = targetBand7
    targetBands[7] = targetBand8



    parameters = snappy.HashMap()
    parameters.put('targetBands', targetBands)

    Indices = GPF.createProduct('BandMaths', parameters, TimeAverage)
    print('Writing the Indices')
    '''WRITING'''
    write(Indices, f'/share/wildfire-3/Sohel/Output/Indices/{Fire_Case}')

    # =======================================================


