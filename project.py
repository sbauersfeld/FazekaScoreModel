import nibabel as nib
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

ALL_DATA = 'Data' # The root directory for our data
NUM_PATIENTS = 8 # The number of patients we want to use

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

def get_train_test_data():
    """
    Get training and test data and labels.

    Returns
    --------------------
    train input     -- numpy array with shape (n,) each entry is a numpy array of the nth patients scans
    test input      -- same as train input but with testing data
    train output    -- numpy array with shape (n,) each entry is the labeled fazeka score for the nth patient
    test output     -- same as train output but with testing data
    """
    all_data_input = np.empty((NUM_PATIENTS), dtype=list)
    all_data_labels = np.zeros((NUM_PATIENTS)) #TODO: Fill in fazeka scores
    i=0
    for s in os.listdir(ALL_DATA):
        if i >= NUM_PATIENTS: # use num_patients to limit the amount of data used
            break

        all_data_input[i] = load_patient_scans(ALL_DATA + '/' + s) # load the data for each patient
        i += 1

    return train_test_split(all_data_input, all_data_labels, test_size=0.2) # split data into training and testing set

def main():
    train_input, test_input, train_output, test_output = get_train_test_data()
    print(np.shape(train_input))
    print(np.shape(train_input[0]))
    # for scan in train_input[0]:
    #     print('The shape is:')
    #     print(np.shape(scan))
    #     plt.imshow(scan, cmap='gray')
    #     plt.show()

if __name__ == '__main__':
    main()