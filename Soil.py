# Required Imports
from osgeo import gdal as gd
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import six

# Import Data
data = gd.Open('/home/chirag/soil_s.tif')
data_raster = data.GetRasterBand(1)
data_array = data_raster.ReadAsArray()

# Required Functions
# Calculates zscore
def calculate_zscore(band_number):
    band_raster = data.GetRasterBand(band_number)
    band_array = band_raster.ReadAsArray()
    band_array_reshaped = band_array.reshape(band_array.shape[0]*band_array.shape[1])

    # Define columns
    columns = ['NDSAI', 'CARB', 'SBL', 'SCI']
    df = pd.DataFrame(band_array_reshaped, columns=[columns[band_number-1]])
    df = df[df[columns[band_number-1]] !=-9999.0]

    #Calculate Zscore
    zscore = pd.DataFrame(stats.zscore(df.values, axis=0, ddof=1), index=df.index, columns=df.columns)
    return zscore
# Plot zscore
# def plot(zscore):
#     sns.distplot(zscore.values, kde=True, color='r', hist_kws={'alpha':0.5})
#     plt.show()

# ZSCORES
ndsai_zscore = calculate_zscore(1)
carb_zscore = calculate_zscore(2)
sbl_zscore = calculate_zscore(3)
sci_zscore = calculate_zscore(4)

# FINAL ZSCORE
final_zscore = (ndsai_zscore['NDSAI'] + carb_zscore['CARB'] + sbl_zscore['SBL'] + sci_zscore['SCI'])

# DATAFRAME
df = pd.DataFrame()
df['NDSAI'] = ndsai_zscore['NDSAI']
df['CARB'] = carb_zscore['CARB']
df['SBL'] = sbl_zscore['SBL']
df['SCI'] = sci_zscore['SCI']

# NEW DATAFRAME
dk = df.copy()
dk[(dk>-3)]=0
dk[(dk<-3)]=1
# print dk.min(), '\n', dk.max()

dk['Score'] = dk.sum(axis=1)
# print dk['Score'].min(), '\n', dk['Score'].max()

# Converting to tiff
def make_df(band_number):
    band_raster = data.GetRasterBand(band_number)
    band_array = band_raster.ReadAsArray()
    band_array_reshaped = band_array.reshape(band_array.shape[0]*band_array.shape[1])

    # Define columns
    columns = ['NDSAI']
    df = pd.DataFrame(band_array_reshaped, columns=[columns[band_number-1]])
    return df

ndsai = make_df(1)
ndsai['Score'] = dk['Score']
ndsai.drop('NDSAI', axis=1, inplace=True)

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

array = df_to_sarray(ndsai)

# Saving to tiff
dst_filename = 'Soil_Score.tiff'
x_pixels = data_array.shape[1]  # number of pixels in x
y_pixels = data_array.shape[0]  # number of pixels in y
driver = gd.GetDriverByName('GTiff')
output_raster = driver.Create(dst_filename,x_pixels, y_pixels, 1, gd.GDT_Float32)
output_raster.GetRasterBand(1).WriteArray(array)

# follow code is adding GeoTranform and Projection
geotrans=data.GetGeoTransform()  #get GeoTranform from existed 'data0'
proj=data.GetProjection() #you can get from a exsited tif or import
output_raster.SetGeoTransform(geotrans)
output_raster.SetProjection(proj)
output_raster.FlushCache()
output_raster=None
