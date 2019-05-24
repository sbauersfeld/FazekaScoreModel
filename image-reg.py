import util
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as scp

import SimpleITK as sitk

def register_image(fixed_image, moving_image):
    fixedImage = sitk.GetImageFromArray(fixed_image)
    movingImage = sitk.GetImageFromArray(moving_image)
    parameterMap = sitk.GetDefaultParameterMap('translation')

    elastixImageFilter = sitk.SimpleElastix()
    # elastixImageFilter.LogToFileOff()
    elastixImageFilter.SetFixedImage(fixedImage)
    elastixImageFilter.SetMovingImage(movingImage)
    elastixImageFilter.SetParameterMap(parameterMap)
    # parameterMapVector = sitk.VectorOfParameterMap()
    # parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
    # parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
    # elastixImageFilter.SetParameterMap(parameterMapVector)
    elastixImageFilter.Execute()

    resultImage = elastixImageFilter.GetResultImage()
    # transformParameterMap = elastixImageFilter.GetTransformParameterMap()
    res = sitk.GetArrayFromImage(resultImage)
    return res

def main():
    template_data = util.load_template_data(util.TEMPLATE_DATA) # load the template data
    num_scans, image_height, image_width = np.shape(template_data) # get the dimensions of the output data
    data_paths = os.listdir(util.ORIGINAL_DATA) # original data paths
    data_paths.sort(key = lambda val: int(val)) # sort the paths so that data is loaded in order of patient ID
    mat_data = np.zeros((1,num_scans,image_height,image_width), dtype=np.float32)
    for i in range(len(data_paths)):
        data = util.load_patient_scans(util.ORIGINAL_DATA + "/" + data_paths[i]) # load the patient data
        reg_data = register_image(template_data, data) # run the elastix regularizer
        if (i == 0):
            mat_data[i] = reg_data
        else: # create the data matrix
            mat_data = np.concatenate((mat_data, np.reshape(reg_data, (1, num_scans, image_height, image_width))))

    # print(mat_data.shape)
    filename = util.PREPROCESSED_DATA
    mat_dict = {}
    mat_dict['data'] = mat_data
    scp.io.savemat(filename, mat_dict)

if __name__ == '__main__':
    main()