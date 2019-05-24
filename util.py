import os
import pydicom
import numpy as np
from scipy.io import loadmat
import nibabel as nib

ALL_DATA = 'Data' # The root directory for our data
PREPROCESSED_DATA = 'Data/Preprocessed'
ORIGINAL_DATA = 'Data/Original'
# NUM_PATIENTS = 1 # The number of patients we want to use
# NUM_CLASSES = 6 # The number of unique fazeka scores
# NUM_SCANS = 12 # TODO: Set this to the correct number of scans we will use per patient
SKIP_BOTTOM = 10
SKIP_TOP = 1
TEMPLATE_BOTTOM = 36
TEMPLATE_TOP = 156
TEMPLATE_ITER = 5
# IMAGE_WIDTH = 230 # TODO: Set this value
# IMAGE_HEIGHT = 256 # TODO: Set this value

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
    dcmImages = dcmImages[SKIP_BOTTOM:len(dcmImages)-SKIP_TOP]
    # print(dcmImages[0].ImagePositionPatient[2])
    dcm_scans = [image.pixel_array for image in dcmImages] # get the pixel array from each dicom image
    dcm_scans = np.array(dcm_scans)
    # dcm_scans = np.moveaxis(dcm_scans, 0, 2) # swap axes for inputting data to conv nets
    return dcm_scans

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

def load_template_data(path):
    img = nib.load(path)
    data = img.get_data()
    _, ext = os.path.splitext(path)
    print(ext)
    if ext == ".nii":
        data = data.T
        data = data[TEMPLATE_BOTTOM:TEMPLATE_TOP:TEMPLATE_ITER]
        data = np.flip(data, 1)
    return data


#     def get_data(data_type):
#     """
#     Get training and test data and labels.

#     Parameters
#     --------------------
#     data type   -- string of either "dcm" or "mat" specifiying which data to load

#     Returns
#     --------------------
#     train input     -- numpy array with shape (n,x,y,m) n training samples for m xy pixel arrays
#     test input      -- same as train input but with testing data
#     train output    -- numpy array with shape (n,q) each entry is the labeled fazeka score for the nth patient in one hot encoding
#     test output     -- same as train output but with testing data
#     """
#     if data_type == 'dcm': # choose which data to load/use
#         data_path = ORIGINAL_DATA
#         load_func = util.load_patient_scans
#     elif data_type == 'mat':
#         data_path = PREPROCESSED_DATA
#         load_func = util.load_processed_data
#     else: 
#         raise Exception("Input to get_data invalid. It must be either \'dcm\' or \'mat\'. It was: {}".format(data_type))

#     all_data_input = np.empty((NUM_PATIENTS), dtype=list)
#     #all_data_input = np.zeros((NUM_PATIENTS, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_SCANS)) #TODO: use this when values known
#     all_data_labels = np.zeros((NUM_PATIENTS)) #TODO: Fill in labeled fazeka scores
#     i=0
#     for s in os.listdir(data_path):
#         if i >= NUM_PATIENTS: # use num_patients to limit the amount of data used
#             break

#         all_data_input[i] = load_func(data_path + '/' + s) # load the data for each patient
#         i += 1

#     # all_data_input = all_data_input.astype(np.float32)
#     train_input, test_input, train_output, test_output = train_test_split(all_data_input, all_data_labels, test_size=0.0) # split data into training and testing set
#     train_output = to_categorical(train_output, num_classes=NUM_CLASSES)
#     test_output = to_categorical(test_output, num_classes=NUM_CLASSES)
#     return train_input, test_input, train_output, test_output