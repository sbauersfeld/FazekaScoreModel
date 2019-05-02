import nibabel as nib
import numpy as np
import pydicom

dcmImage = pydicom.dcmread('Data/101_8_30_15/IM-0065-0001.dcm')
dcm_scan = dcmImage.pixel_array
print('The shape is:')
print(np.shape(dcm_scan))