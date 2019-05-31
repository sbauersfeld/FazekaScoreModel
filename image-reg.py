import util
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as scp

import SimpleITK as sitk

# TODO: determine best parameters, figure out patient 151
def register_image(fixed_image, moving_image):
    fixedImage = sitk.GetImageFromArray(fixed_image)
    movingImage = sitk.GetImageFromArray(moving_image)
    parameterMap = sitk.GetDefaultParameterMap('translation')
    # parameterMap['AutomaticTransformInitialization'] = ["true"]
    # parameterMap['Transform'] = ['BSplineTransform']
    # parameterMap['MaximumNumberOfIterations'] = ['512']
    # sitk.PrintParameterMap(parameterMap)

    elastixImageFilter = sitk.SimpleElastix()
    # elastixImageFilter.LogToConsoleOn()
    elastixImageFilter.SetFixedImage(fixedImage)
    elastixImageFilter.SetMovingImage(movingImage)
    # elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap('rigid'))
    elastixImageFilter.SetParameterMap(parameterMap)
    # parameterMapVector = sitk.VectorOfParameterMap()
    # parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
    # parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
    # parameterMapVector['MaximumNumberOfIterations'] = ['1024']
    # elastixImageFilter.SetParameterMap(parameterMapVector)
    elastixImageFilter.Execute()

    resultImage = elastixImageFilter.GetResultImage()
    # transformParameterMap = elastixImageFilter.GetTransformParameterMap()
    res = sitk.GetArrayFromImage(resultImage)
    return res

def main():
    template_data = util.load_template_data(util.TEMPLATE_DATA) # load the template data
    num_scans, image_height, image_width = np.shape(template_data) # get the dimensions of the output data
    data_paths = os.listdir(util.NORMALIZED_DATA_Z_SCORES) # original data paths
    data_paths.sort(key = lambda val: int(val.replace(".mat", ""))) # sort the paths so that data is loaded in order of patient ID
    mat_data = np.zeros((1,num_scans,image_height,image_width), dtype=np.float32)
    num_patients = len(data_paths)
    print("Processing data for {} patients".format(num_patients))
    for i in range(num_patients):
        print("Iteration: ", i)
        if i == 40: # this patient has issues :( we need to add it to our data manually
            data = util.load_processed_data(util.NORMALIZED_DATA_Z_SCORES + "/" + data_paths[i])[0] # load the patient data
            data = data.T
            data = np.fliplr(data)
            a,b,c = np.shape(data)
            y = (b - image_height)/2
            x = (c - image_width)/2
            reg_data = data[a-num_scans:,y:b-y,x:c-x] # this is a hack, just resizing to fit dimensions
            # util.multi_slice_subplot(reg_data)
            # plt.show()
        else:
            data = util.load_processed_data(util.NORMALIZED_DATA_Z_SCORES + "/" + data_paths[i])[0] # load the patient data
            data = data.T
            data = np.fliplr(data)
            # for i range(len(data)):
                # image_slic = seg.slic(data[i],compactness=0.001,n_segments=1000,multichannel=False,slic_zero=True)
                # data[i] = color.label2rgb(image_slic, data[i], kind='avg')
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