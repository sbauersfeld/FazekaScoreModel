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

def main():

    data = util.load_processed_data(util.PREPROCESSED_DATA)
    print(np.shape(data))
    util.multi_slice_subplot(data[0])
    util.multi_slice_subplot(data[5])
    plt.show()
    # train_input, test_input, train_output, test_output = get_data("dcm")

    # inum = 15

    # image_slic = seg.slic(train_input[0][:,:,inum],compactness=0.001,n_segments=1000,multichannel=False,slic_zero=True)
    # image_slic2 = color.label2rgb(image_slic, train_input[0][:,:,inum], kind='avg')
    
    # out = seg.mark_boundaries(image_slic2, image_slic, color=(0,0,1))
    # plt.imshow(image_slic2,cmap='gray',interpolation=None)
    # plt.imshow(out, interpolation=None,alpha=0.25)
    # plt.show()

if __name__ == '__main__':
    main()