import nibabel as nib
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import os

training_data_paths = ['Data/101_8_30_15', 
                       'Data/106_9_13_15', 
                       'Data/108_9_18_15',
                       'Data/111_9_22_15',
                       'Data/116_9_29_15',
                       'Data/118_10_1_15']

testing_data_paths = ['Data/122_10_20_15',
                      'Data/128_10_27_15']

num_train = len(training_data_paths)
num_test = len(testing_data_paths)

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
    train output    -- numpy array with shape (n,) each entry is the labeled fazeka score for the nth patient
    test input      -- same as train input but with testing data
    test output     -- same as train output but with testing data
    """
    train_input = np.empty((num_train), dtype=list) # create the numpy arrays
    train_output = np.zeros((num_train))            # TODO: Fill in fazeka scores
    test_input = np.empty((num_test), dtype=list)
    test_output = np.zeros((num_test))
    for i in range(num_train):
        train_input[i] = load_patient_scans(training_data_paths[i]) # load all of a patients scans
    for j in range(num_test):
        test_input[j] = load_patient_scans(testing_data_paths[j])
    return train_input, train_output, test_input, test_output

def main():
    train_input, train_output, test_input, test_output = get_train_test_data()
    print(np.shape(train_input))
    print(np.shape(train_input[0]))
    # for scan in train_input[0]:
    #     print('The shape is:')
    #     print(np.shape(scan))
    #     plt.imshow(scan, cmap='gray')
    #     plt.show()

if __name__ == '__main__':
    main()