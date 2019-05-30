import os
import pydicom
import numpy as np
from scipy.io import loadmat
import nibabel as nib
import matplotlib.pyplot as plt
import xlrd

ALL_DATA = 'Data' # The root directory for our data
PREPROCESSED_DATA = 'Data/Preprocessed/regularized_data'
ORIGINAL_DATA = 'Data/Original'
TEMPLATE_DATA = "Data/GG-366-FLAIR-1.0mm.nii"
LABEL_DATA = 'Data/Scores/All_Fazekas_Data.xlsx'
SKIP_BOTTOM = 10
SKIP_TOP = 1
TEMPLATE_BOTTOM = 36
TEMPLATE_TOP = 156
TEMPLATE_ITER = 10

def load_patient_scans(data_path, skip_bottom=SKIP_BOTTOM, skip_top=SKIP_TOP):
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
    dcmImages = dcmImages[skip_bottom:len(dcmImages)-skip_top]
    # print(dcmImages[0].ImagePositionPatient[2])
    dcm_scans = [image.pixel_array for image in dcmImages] # get the pixel array from each dicom image
    dcm_scans = np.asarray(dcm_scans, dtype=np.float32)
    # dcm_scans = np.moveaxis(dcm_scans, 0, 2) # swap axes for inputting data to conv nets
    return dcm_scans

#RESEARCHER OPTIONS: "1","2", or any other string
#Researcher 1 or 2 will only take the scores from that researcher, otherwise will take the averaged scores
#TYPE OPTIONS: "peri", "deep", ""
# peri will take periventricular score, deep will take deep score, otherwise will take the combined scores (0-6)
def load_patient_labels(data_path, researcher, type):
    wb = xlrd.open_workbook(data_path)
    sheet = wb.sheet_by_index(0)
    if (researcher=="1"):
        if(type=="peri"):
            labels = np.array(sheet.col_values(1,1))
        elif(type=="deep"):
            labels = np.array(sheet.col_values(2,1))
        else:
            labels = np.array(sheet.col_values(3,1))
    elif (researcher=="2"):
        if(type=="peri"):
            labels = np.array(sheet.col_values(4,1))
        elif(type=="deep"):
            labels = np.array(sheet.col_values(5,1))
        else:
            labels = np.array(sheet.col_values(6,1))
    else:
        if(type=="peri"):
            labels = np.array(sheet.col_values(7,1))
        elif(type=="deep"):
            labels = np.array(sheet.col_values(8,1))
        else:
            labels = np.array(sheet.col_values(9,1))
    return labels


def load_processed_data(data_path):
    """
    Load patient data.
        
    Parameters
    --------------------
        data path    -- string formatted path to the folder containing the patient data
    
    Returns
    --------------------
        data        -- numpy array pixel arrays with shape (m,n,x,y), 
                       for each of the m patients, get one x,y pixel 
                       array for each of the n regularized images
    """
    data = loadmat(data_path)['data'] # load the data from the mat file
    return data

def load_template_data(path):
    img = nib.load(path)
    data = img.get_data()
    _, ext = os.path.splitext(path)
    print(ext)
    if ext == ".nii": # this is the template version we will be using
        data = data.T # it needs to be formatted correctly
        data = data[TEMPLATE_BOTTOM:TEMPLATE_TOP:TEMPLATE_ITER] # select the slices we want to use
        data = np.fliplr(data)
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


def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

def multi_slice_viewer(volume): # this lets us look through all slices in a 3D stack
    """
    Plot 3D image. View difference slices with keys j and k
        
    Parameters
    --------------------
        volume    -- 3D image to view, with slices along first axis
    """
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

def multi_slice_subplot(data): # plot all slices on a subplot
    """
    Plot 3D image, with each slice on a separate subplot
        
    Parameters
    --------------------
        volume    -- 3D image to view, with slices along first axis
    """
    n,y,x = np.shape(data)
    x_lim = 4
    y_lim = int(np.ceil(n/x_lim))
    f, axarr = plt.subplots(y_lim,x_lim)
    index = 0
    for i in range(y_lim):
        for j in range(x_lim):
            axarr[i,j].imshow(data[index], cmap='gray')
            index += 1
            if index >= n:
                break
        if index >= n:
            break

# from skimage import segmentation as seg
# import skimage.color as color
# from skimage.segmentation import mark_boundaries
# from skimage import io

# def segment():
    # inum = 15

    # image_slic = seg.slic(train_input[0][:,:,inum],compactness=0.001,n_segments=1000,multichannel=False,slic_zero=True)
    # image_slic2 = color.label2rgb(image_slic, train_input[0][:,:,inum], kind='avg')
    
    # out = seg.mark_boundaries(image_slic2, image_slic, color=(0,0,1))
    # plt.imshow(image_slic2,cmap='gray',interpolation=None)
    # plt.imshow(out, interpolation=None,alpha=0.25)
