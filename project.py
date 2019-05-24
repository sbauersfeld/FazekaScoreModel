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

def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

def multi_slice_viewer(volume):
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] // 2
    ax.imshow(volume[ax.index], cmap='gray')
    fig.canvas.mpl_connect('key_press_event', process_key)

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
        print(ax.index)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()

def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])

def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])

def main():
    data = util.load_template_data("Data/GG-366-FLAIR-1.0mm.nii")
    print(data.shape)
    multi_slice_viewer(data)

    pat_data = util.load_patient_scans("Data/Original/101_8_30_15")
    print(np.shape(pat_data))
    multi_slice_viewer(pat_data)
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