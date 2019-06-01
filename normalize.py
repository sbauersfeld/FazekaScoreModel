import util
import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import scipy as scp
from intensity_normalization.normalize import zscore

def main():
#util.generate_nifti_images(util.ORIGINAL_DATA) if the data is already in the nifti folder, unnecessary to call again
    fols = os.listdir(util.NIFTI_DATA)
    i = 1
    for folname in fols:
        for fname in os.listdir(os.path.join(util.NIFTI_DATA,folname)):
            picnift = nib.load(os.path.join(util.NIFTI_DATA,folname, fname))
            normalized = zscore.zscore_normalize(picnift) #this is the most basic normalization
            img_data = normalized.get_fdata()
            s = img_data.shape
            img_data = img_data[:,:, util.SKIP_BOTTOM:(s[2] - util.SKIP_TOP)]    
            #img_data is type numpy ndarray   
            s = img_data.shape
            mat1 = np.zeros((1,s[0],s[1],s[2]), dtype=np.float32)
            if (i==1):
                print(s)
                i = i+1
                slice_0 = img_data[int(np.floor(s[0]/2)), :, :]
                slice_1 = img_data[:, int(np.floor(s[1]/2)), :]
                slice_2 = img_data[:, :, int(np.floor(s[2]/2))]
                util.show_slices([slice_0, slice_1, slice_2])
                plt.suptitle("Center slices for 1st patient")  # doctest: +SKIP
                plt.show()
            mat1[0] = img_data
            filename = folname
            mat_dict = {}
            mat_dict['data'] = mat1
            #scp.io.savemat(os.path.join(util.NIFTI_DATA,filename), mat_dict)

if __name__ == '__main__':
    main()