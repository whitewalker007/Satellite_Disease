# Required Imports
from osgeo import gdal as gd
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import six

# Import Data
Data = gd.Open('/home/chirag/JUNAGADH/junagadh_ndwi.tif')
data_raster = Data.GetRasterBand(1)
data_array = data_raster.ReadAsArray()

# Required Functions
# Calculates zscore
def calculate_zscore_1(band_number):
    data = gd.Open('/home/chirag/JUNAGADH/junagadh_ndwi.tif')
    band_raster = data.GetRasterBand(band_number)
    band_array = band_raster.ReadAsArray()
    band_array_reshaped = band_array.reshape(band_array.shape[0]*band_array.shape[1])

    # Define columns
    columns = ['NDWI', 'MNDWI', 'GVMI']
    df = pd.DataFrame(band_array_reshaped, columns=[columns[band_number-1]])
    df = df[df[columns[band_number-1]] !=-9999.0]

    #Calculate Zscore
    zscore = pd.DataFrame(stats.zscore(df.values, axis=0, ddof=1), index=df.index, columns=df.columns)
    return zscore

def calculate_zscore_2(band_number):
    data = gd.Open('/home/chirag/JUNAGADH/junagadh_mndwi.tif')
    band_raster = data.GetRasterBand(band_number)
    band_array = band_raster.ReadAsArray()
    band_array_reshaped = band_array.reshape(band_array.shape[0]*band_array.shape[1])

    # Define columns
    columns = ['NDWI', 'MNDWI', 'GVMI']
    df = pd.DataFrame(band_array_reshaped, columns=[columns[band_number]])
    df = df[df[columns[band_number]] !=-9999.0]

    #Calculate Zscore
    zscore = pd.DataFrame(stats.zscore(df.values, axis=0, ddof=1), index=df.index, columns=df.columns)
    return zscore

def calculate_zscore_3(band_number):
    data = gd.Open('/home/chirag/JUNAGADH/junagadh_gvmi.tif')
    band_raster = data.GetRasterBand(band_number)
    band_array = band_raster.ReadAsArray()
    band_array_reshaped = band_array.reshape(band_array.shape[0]*band_array.shape[1])

    # Define columns
    columns = ['NDWI', 'MNDWI', 'GVMI']
    df = pd.DataFrame(band_array_reshaped, columns=[columns[band_number+1]])
    df = df[df[columns[band_number+1]] !=-9999.0]

    #Calculate Zscore
    zscore = pd.DataFrame(stats.zscore(df.values, axis=0, ddof=1), index=df.index, columns=df.columns)
    return zscore
# Plot zscore
# def plot(zscore):
#     sns.distplot(zscore.values, kde=True, color='r', hist_kws={'alpha':0.5})
#     plt.show()

# ZSCORES
ndwi_zscore = calculate_zscore_1(1)
mndwi_zscore = calculate_zscore_2(1)
gvmi_zscore = calculate_zscore_3(1)

# FINAL ZSCORE
final_zscore = (ndwi_zscore['NDWI'] + mndwi_zscore['MNDWI'] + gvmi_zscore['GVMI'])

# DATAFRAME
df = pd.DataFrame()
df['NDWI'] = ndwi_zscore['NDWI']
df['MNDWI'] = mndwi_zscore['MNDWI']
df['GVMI'] = gvmi_zscore['GVMI']

# NEW DATAFRAME
dk = df.copy()
dk[(dk>-3)]=0
dk[(dk<-3)]=1
# print dk.min(), '\n', dk.max()

dk['Score'] = dk.sum(axis=1)
# print dk['Score'].min(), '\n', dk['Score'].max()

# Converting to tiff
def make_df(band_number):
    band_raster = Data.GetRasterBand(band_number)
    band_array = band_raster.ReadAsArray()
    band_array_reshaped = band_array.reshape(band_array.shape[0]*band_array.shape[1])

    # Define columns
    columns = ['NDWI']
    df = pd.DataFrame(band_array_reshaped, columns=[columns[band_number-1]])
    return df

ndwi = make_df(1)
ndwi['Score'] = dk['Score']
ndwi.drop('NDWI', axis=1, inplace=True)

# DataFrame to Array
def df_to_sarray(df):
    """
    Convert a pandas DataFrame object to a numpy structured array.
    This is functionally equivalent to but more efficient than
    np.array(df.to_array())

    :param df: the data frame to convert
    :return: a numpy structured array representation of df
    """

    v = df.values
    cols = df.columns

    if six.PY2:  # python 2 needs .encode() but 3 does not
        types = [(cols[i].encode(), df[k].dtype.type) for (i, k) in enumerate(cols)]
    else:
        types = [(cols[i], df[k].dtype.type) for (i, k) in enumerate(cols)]
    dtype = np.dtype(types)
    z = np.zeros(v.shape[0], dtype)
    for (i, k) in enumerate(z.dtype.names):
        z[k] = v[:, i]
    return z.reshape(data_array.shape[0], data_array.shape[1])

array = df_to_sarray(ndwi)

# Saving to tiff
dst_filename = 'Water_Score.tiff'
x_pixels = data_array.shape[1]  # number of pixels in x
y_pixels = data_array.shape[0]  # number of pixels in y
driver = gd.GetDriverByName('GTiff')
output_raster = driver.Create(dst_filename,x_pixels, y_pixels, 1, gd.GDT_Float32)
output_raster.GetRasterBand(1).WriteArray(array)

# follow code is adding GeoTranform and Projection
geotrans=Data.GetGeoTransform()  #get GeoTranform from existed 'data0'
proj=Data.GetProjection() #you can get from a exsited tif or import
output_raster.SetGeoTransform(geotrans)
output_raster.SetProjection(proj)
output_raster.FlushCache()
output_raster=None
