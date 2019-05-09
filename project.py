import nibabel as nib
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from scipy.io import loadmat

ALL_DATA = 'Data' # The root directory for our data
PREPROCESSED_DATA = 'Data/Preprocessed'
ORIGINAL_DATA = 'Data/Original'
NUM_PATIENTS = 1 # The number of patients we want to use

def load_patient_scans(data_path):
    """
    Load patient data.
        
    Parameters
    --------------------
        data path    -- string formatted path to the folder containing the patient data
    
    Returns
    --------------------
        dcm scans    -- numpy array pixel arrays with shape (n,x,y), 
                        one pixel array for each of the n patient images
    """
    dcmImages = [pydicom.read_file(data_path + '/' + s) for s in os.listdir(data_path)] # read the dicom image
    dcmImages.sort(key = lambda image: image.ImagePositionPatient[2]) # ensure the list is sorted in z-dimension (might not be needed)
    dcm_scans = [image.pixel_array for image in dcmImages] # get the pixel array from each dicom image
    return dcm_scans

def load_processed_data(data_path):
    """
    Load patient data.
        
    Parameters
    --------------------
        data path    -- string formatted path to the folder containing the patient data
    
    Returns
    --------------------
        data        -- numpy array pixel arrays with shape (n,x,y), 
                        one pixel array for each of the n patient images
    """
    data = loadmat(data_path)['stack1'] # load the data TODO: change this to 'stack'
    return np.moveaxis(data, -1, 0) # swap axes for consistency with original data

def get_data(data_type):
    """
    Get training and test data and labels.

    Parameters
    --------------------
    data type   -- string of either "dcm" or "mat" specifiying which data to load

    Returns
    --------------------
    train input     -- numpy array with shape (n,) each entry is a numpy array of the nth patients scans
    test input      -- same as train input but with testing data
    train output    -- numpy array with shape (n,) each entry is the labeled fazeka score for the nth patient
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
    all_data_labels = np.zeros((NUM_PATIENTS)) #TODO: Fill in fazeka scores
    i=0
    for s in os.listdir(data_path):
        if i >= NUM_PATIENTS: # use num_patients to limit the amount of data used
            break

        all_data_input[i] = load_func(data_path + '/' + s) # load the data for each patient
        i += 1

    train_input, test_input, train_output, test_output = train_test_split(all_data_input, all_data_labels, test_size=0.0) # split data into training and testing set
    train_output = to_categorical(train_output, num_classes=12)
    test_output = to_categorical(test_output, num_classes=12)
    return train_input, test_input, train_output, test_output

def main():
    train_input, test_input, train_output, test_output = get_data("mat")
    print(np.shape(train_input))
    print(np.shape(test_input))
    print(np.shape(train_input[0]))
    print(np.shape(train_output))
    for scan in train_input[0]:
        print('The shape is:')
        print(np.shape(scan))
        plt.imshow(scan, cmap='gray')
        plt.show()

if __name__ == '__main__':
    main()