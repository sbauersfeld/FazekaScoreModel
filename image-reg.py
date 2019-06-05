import util
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as scp

import SimpleITK as sitk

# this function peforms image registration with simple elastix
def register_image(fixed_image, moving_image):
    fixedImage = sitk.GetImageFromArray(fixed_image)
    movingImage = sitk.GetImageFromArray(moving_image)
    parameterMap = sitk.GetDefaultParameterMap('translation') # translation transformation has the best results

    elastixImageFilter = sitk.SimpleElastix()
    elastixImageFilter.SetFixedImage(fixedImage)
    elastixImageFilter.SetMovingImage(movingImage)
    elastixImageFilter.SetParameterMap(parameterMap)
    elastixImageFilter.Execute() # transforms the moving image to align with the coordinate system of the fixed image

    resultImage = elastixImageFilter.GetResultImage()
    res = sitk.GetArrayFromImage(resultImage)
    return res

def main():
    template_data = util.load_template_data(util.TEMPLATE_DATA) # load the template data
    num_scans, image_height, image_width = np.shape(template_data) # get the dimensions of the output data
    data_paths = os.listdir(util.NORMALIZED_DATA_Z_SCORES) # original data paths
    data_paths.sort(key = lambda val: int(val.replace(".mat", ""))) # sort the paths so that data is loaded in order of patient ID and will align with labels
    mat_data = np.zeros((1,num_scans,image_height,image_width), dtype=np.float32)
    num_patients = len(data_paths)
    print("Processing data for {} patients".format(num_patients))
    for i in range(num_patients):
        if i % 10 == 0: # track progress
            print("Iteration: ", i)

        # this patient has MRIs that are not within the same frame of refrence as other patients 
        # and cannot be registered properly :( we need to add it to our data manually 
        # consider removing this sample when training CNN
        if i == 40:
            data = util.load_processed_data(util.NORMALIZED_DATA_Z_SCORES + "/" + data_paths[i])[0]
            data = data.T
            data = np.fliplr(data)
            a,b,c = np.shape(data)
            y = (b - image_height)/2
            x = (c - image_width)/2
            reg_data = data[a-num_scans:,y:b-y,x:c-x] # this is a hack, just resizing to fit dimensions
        else: # every other patient's registration works correctly
            data = util.load_processed_data(util.NORMALIZED_DATA_Z_SCORES + "/" + data_paths[i])[0] # load the normalized patient data
            data = data.T
            data = np.fliplr(data)
            reg_data = register_image(template_data, data) # run the elastix regularizer
        if i == 0:
            mat_data[i] = reg_data
        else: # create the data matrix
            mat_data = np.concatenate((mat_data, np.reshape(reg_data, (1, num_scans, image_height, image_width))))

    print("Regularized data shape is: {}".format(mat_data.shape))
    filename = util.PREPROCESSED_Z_SCORES
    mat_dict = {}
    mat_dict['data'] = mat_data
    scp.io.savemat(filename, mat_dict)

if __name__ == '__main__':
    main()