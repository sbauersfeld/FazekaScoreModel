import util

import os
# import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# import keras
# from keras.utils import to_categorical
# from keras.models import Sequential,Input,Model
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D, MaxPooling2D, Conv3D, MaxPooling3D, GlobalMaxPooling3D
# from keras.layers.normalization import BatchNormalization
# from keras.layers.advanced_activations import LeakyReLU

# from skimage import segmentation as seg
# import skimage.color as color
# from skimage.segmentation import mark_boundaries
# from skimage import io

ALL_DATA = 'Data' # The root directory for our data
PREPROCESSED_DATA = 'Data/Preprocessed'
ORIGINAL_DATA = 'Data/Original'
NUM_PATIENTS = 1 # The number of patients we want to use
NUM_CLASSES = 6 # The number of unique fazeka scores
NUM_SCANS = 12 # TODO: Set this to the correct number of scans we will use per patient
IMAGE_WIDTH = 230 # TODO: Set this value
IMAGE_HEIGHT = 256 # TODO: Set this value

def main():
    data = util.load_template_data("Data/GG-366-FLAIR-1.0mm.nii")
    print(data.shape)
    util.multi_slice_viewer(data)

    pat_data = util.load_patient_scans("Data/Original/101_8_30_15")
    print(np.shape(pat_data))
    util.multi_slice_viewer(pat_data)
    plt.show()


    # train_input, test_input, train_output, test_output = get_data("dcm")

    # inum = 15

    # image_slic = seg.slic(train_input[0][:,:,inum],compactness=0.001,n_segments=1000,multichannel=False,slic_zero=True)
    # image_slic2 = color.label2rgb(image_slic, train_input[0][:,:,inum], kind='avg')
    
    # out = seg.mark_boundaries(image_slic2, image_slic, color=(0,0,1))
    # plt.imshow(image_slic2,cmap='gray',interpolation=None)
    # plt.imshow(out, interpolation=None,alpha=0.25)
    # plt.show()

    # print(np.shape(train_output))
    # for patient in train_input:
    #     print(patient.shape)
    #     f, axarr = plt.subplots(4,3)
    #     index = 0
    #     for i in range(4):
    #         for j in range(3):
    #             axarr[i,j].imshow(patient[:,:,index], cmap='gray')
    #             index += 1
    #             if index >= NUM_SCANS:
    #                 break

    # plt.show()

if __name__ == '__main__':
    main()