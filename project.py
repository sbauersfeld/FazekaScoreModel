import os
import nibabel as nib
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import keras
from keras.utils import to_categorical
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

ALL_DATA = 'Data' # The root directory for our data
PREPROCESSED_DATA = 'Data/Preprocessed'
ORIGINAL_DATA = 'Data/Original'
NUM_PATIENTS = 8 # The number of patients we want to use
NUM_CLASSES = 6 # The number of unique fazeka scores
NUM_SCANS = 12 # TODO: Set this to the correct number of scans we will use per patient
IMAGE_WIDTH = 230 # TODO: Set this value
IMAGE_HEIGHT = 256 # TODO: Set this value

def load_patient_scans(data_path):
    """
    Load patient data.
        
    Parameters
    --------------------
        data path    -- string formatted path to the folder containing the patient data
    
    Returns
    --------------------
        dcm scans    -- numpy array pixel arrays with shape (x,y,n), 
                        one pixel array for each of the n patient images
    """
    dcmImages = [pydicom.read_file(data_path + '/' + s) for s in os.listdir(data_path)] # read the dicom image
    dcmImages.sort(key = lambda image: image.ImagePositionPatient[2]) # ensure the list is sorted in z-dimension (might not be needed)
    dcmImages = dcmImages[len(dcmImages)-NUM_SCANS-4:len(dcmImages)-4]
    dcm_scans = [image.pixel_array for image in dcmImages] # get the pixel array from each dicom image
    return np.moveaxis(dcm_scans, 0, 2) # swap axes for inputting data to conv nets

def load_processed_data(data_path):
    """
    Load patient data.
        
    Parameters
    --------------------
        data path    -- string formatted path to the folder containing the patient data
    
    Returns
    --------------------
        data        -- numpy array pixel arrays with shape (x,y,n), 
                        one pixel array for each of the n patient images
    """
    data = loadmat(data_path)['stack'] # load the data TODO: change this to 'stack'
    return data

def get_data(data_type):
    """
    Get training and test data and labels.

    Parameters
    --------------------
    data type   -- string of either "dcm" or "mat" specifiying which data to load

    Returns
    --------------------
    train input     -- numpy array with shape (n,x,y,m) n training samples for m xy pixel arrays
    test input      -- same as train input but with testing data
    train output    -- numpy array with shape (n,q) each entry is the labeled fazeka score for the nth patient in one hot encoding
    test output     -- same as train output but with testing data
    """
    if data_type == 'dcm': # choose which data to load/use
        data_path = ORIGINAL_DATA
        load_func = load_patient_scans
    elif data_type == 'mat':
        data_path = PREPROCESSED_DATA
        load_func = load_processed_data
    else: 
        raise Exception("Input to get_data invalid. It must be either \'dcm\' or \'mat\'. It was: {}".format(data_type))

    all_data_input = np.empty((NUM_PATIENTS), dtype=list)
    #all_data_input = np.zeros((NUM_PATIENTS, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_SCANS)) #TODO: use this when values known
    all_data_labels = np.zeros((NUM_PATIENTS)) #TODO: Fill in labeled fazeka scores
    i=0
    for s in os.listdir(data_path):
        if i >= NUM_PATIENTS: # use num_patients to limit the amount of data used
            break

        all_data_input[i] = load_func(data_path + '/' + s) # load the data for each patient
        i += 1

    train_input, test_input, train_output, test_output = train_test_split(all_data_input, all_data_labels, test_size=0.0) # split data into training and testing set
    train_output = to_categorical(train_output, num_classes=NUM_CLASSES)
    test_output = to_categorical(test_output, num_classes=NUM_CLASSES)
    return train_input, test_input, train_output, test_output

def main():
    train_input, test_input, train_output, test_output = get_data("dcm")
    # print(np.shape(train_input))
    # print(np.shape(test_input))
    # print(np.shape(train_input[0]))
    # print(np.shape(train_output))
    for patient in train_input:
        print(patient.shape)
        f, axarr = plt.subplots(4,3)
        index = 0
        for i in range(4):
            for j in range(3):
                axarr[i,j].imshow(patient[:,:,index], cmap='gray')
                index += 1
                if index >= NUM_SCANS:
                    break

    plt.show()

    # for i in range(NUM_SCANS):
    #     plt.imshow(train_input[1][:,:,i], cmap='gray')
    #     plt.show()

    # set up conv net
    # batch_size = 10
    # epochs = 20
    # fazeka_model = Sequential()
    # fazeka_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(IMAGE_HEIGHT,IMAGE_WIDTH,NUM_SCANS),padding='same'))
    # fazeka_model.add(LeakyReLU(alpha=0.1))
    # fazeka_model.add(MaxPooling2D((2, 2),padding='same'))
    # fazeka_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
    # fazeka_model.add(LeakyReLU(alpha=0.1))
    # fazeka_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    # fazeka_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
    # fazeka_model.add(LeakyReLU(alpha=0.1))                  
    # fazeka_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    # fazeka_model.add(Flatten())
    # fazeka_model.add(Dense(128, activation='linear'))
    # fazeka_model.add(LeakyReLU(alpha=0.1))                  
    # fazeka_model.add(Dense(NUM_CLASSES, activation='softmax'))

    # # compile the model
    # fazeka_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
    # fazeka_model.summary()

    # # train the model
    # fashion_train = fazeka_model.fit(train_input, train_output, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))

    # # test the model
    # test_eval = fazeka_model.evaluate(test_input, test_output, verbose=0)
    # print('Test loss:', test_eval[0])
    # print('Test accuracy:', test_eval[1])   

    # for scan in train_input[0]:
    #     print('The shape is:')
    #     print(np.shape(scan))
    #     plt.imshow(scan, cmap='gray')
    #     plt.show()

if __name__ == '__main__':
    main()