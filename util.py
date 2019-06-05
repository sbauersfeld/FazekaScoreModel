import os
import pydicom
import numpy as np
from scipy.io import loadmat
import nibabel as nib
import matplotlib.pyplot as plt
import xlrd
import dicom2nifti

# these are the default paths and settings used
ALL_DATA = 'Data' # The root directory for our data
PREPROCESSED_DATA = 'Data/Preprocessed/regularized_data'
NORMALIZED_DATA_Z_SCORES = 'Data/Normalized/normalized_z_scores'
NORMALIZED_DATA_FCM = 'Data/Normalized/normalized_fcm'
PREPROCESSED_Z_SCORES = 'Data/Preprocessed/regularized_z_scores'
PREPROCESSED_FCM = 'Data/Preprocessed/regularized_fcm'
PREPROCESSED_SEGMENTED = 'Data/Preprocessed/segmented'
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
    dcm_scans = [image.pixel_array for image in dcmImages] # get the pixel array from each dicom image
    dcm_scans = np.asarray(dcm_scans, dtype=np.float32)
    return dcm_scans

def generate_nifti_images(data_path, skip_bottom=SKIP_BOTTOM, skip_top=SKIP_TOP):
    dirname = data_path
    fols = os.listdir(dirname)
    for folname in fols:
        path = os.path.join(NIFTI_DATA,folname)
        if (os.path.isdir(path) == False):
            os.mkdir(path)
        dicom2nifti.convert_directory(os.path.join(dirname,folname), path, compression=True, reorient=True)
        files = os.listdir(os.path.join(dirname,folname))
        ds_list.append([pydicom.read_file(os.path.join(dirname, folname,file))] for file in files)

def show_slices(slices):
   # Function to display row of nifti image slices 
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")

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

def load_template_data(path, template_bottom=TEMPLATE_BOTTOM, template_top=TEMPLATE_TOP, template_iter=TEMPLATE_ITER):
    """
    Load template data to be used as the fixed image in image registration.

    Parameters
    --------------------
        path            -- filepath to the template data 
        template_bottom -- the first template scan to include
        template_top    -- the last template scan to include
        template_iter   -- determines how many scans are included: total is floor((template_top-template_bottom)/template_iter)
    
    Returns
    --------------------
        data        -- numpy array containing the template data
    """
    img = nib.load(path)
    data = img.get_data()
    _, ext = os.path.splitext(path)
    if ext == ".nii": # this is the template version we will be using
        data = data.T # it needs to be formatted correctly
        data = data[template_bottom:TEMPLATE_TOP:TEMPLATE_ITER] # select the slices we want to use
        data = np.fliplr(data)
    return data

def multi_slice_subplot(data): # plot all MRI slices on a subplot
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

# The following code is adapted from "DataCamp’s Viewing 3D Volumetric Data With Matplotlib Tutorial"
# It is publicly available online at: https://www.datacamp.com/community/tutorials/matplotlib-3d-volumetric-data
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
    ax.index = (ax.index - 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])

def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])
