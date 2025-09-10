'''PRE-PROCESSING'''

import sys
sys.path.append('/home/sohel/.snap/snap-python')
import esa_snappy

from os.path import join
import subprocess
from glob import iglob
from zipfile import ZipFile
import numpy as np
import pandas as pd
import esa_snappy
from esa_snappy import jpy, GPF, ProductIO
import matplotlib.pyplot as plt

#set target folder and extract metadata
Fire_data = pd.read_csv('/share/wildfire-3/Sohel/Fire_Data.csv')

Fire_Cases = []

Fire_Cases = ["Lithakia"]

for Fire_Case in Fire_Cases:
    row = Fire_data[Fire_data['Fire_Case'] == Fire_Case]
    #Load the files
    product_path = f'/share/wildfire-3/Sohel/Output/Image/{Fire_Case}/'
    input_S1_files = sorted(list(iglob(join(product_path, '*.dim'), recursive=False)))

    s1_data = []  
    for i in input_S1_files:
        s1_read = esa_snappy.ProductIO.readProduct(i)
        s1_data.append(s1_read)   

    def read(filename):
        return ProductIO.readProduct(filename)

    def write(product, filename):
        ProductIO.writeProduct(product, filename, "BEAM-DIMAP")

    def subset(product, geom):
        parameters = esa_snappy.HashMap()
        parameters.put('geoRegion', geom)
        parameters.put('copyMetadata', True)
        return esa_snappy.GPF.createProduct('Subset', parameters, product)

    '''COREGISTRATION: CREATE STACK'''
    def create_stack(product):
        parameters = esa_snappy.HashMap()
        parameters.put('extent','Master')
        parameters.put('initialOffsetMethod', 'Product Geolocation')
        
        return esa_snappy.GPF.createProduct("CreateStack", parameters, product)

    '''MULTITEMPORAL SPECKLE FILTER'''

    def multitemporal_filter(product):
        parameters = esa_snappy.HashMap()
        parameters.put('filter', 'Lee')
        parameters.put('filterSizeX', 5)
        parameters.put('filterSizeY', 5)
        return esa_snappy.GPF.createProduct('Multi-Temporal-Speckle-Filter', parameters, product)


    '''GLCM'''
    def GLCM(product):
        parameters = esa_snappy.HashMap()
        parameters.put('windowSizeStr', '11x11')
        parameters.put('outputContrast', False)
        parameters.put('outputASM', False)
        parameters.put('outputEnergy', False)
        parameters.put('outputHomogeneity', False)
        parameters.put('outputMAX', False)
        return esa_snappy.GPF.createProduct('GLCM', parameters, product)

    '''Process'''
    stack = create_stack(s1_data)
    filtered = multitemporal_filter(stack)

    bands = filtered.getBandNames()
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

    targetBands = jpy.array('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor', 4)
    targetBands[0] = targetBand1
    targetBands[1] = targetBand2
    targetBands[2] = targetBand3
    targetBands[3] = targetBand4

    parameters = esa_snappy.HashMap()
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



    parameters = esa_snappy.HashMap()
    parameters.put('targetBands', targetBands)

    Indices = GPF.createProduct('BandMaths', parameters, TimeAverage)
    print('Writing the Indices')
    '''WRITING'''
    write(Indices, f'/share/wildfire-3/Sohel/Output/Indices/{Fire_Case}_2')

    # =======================================================