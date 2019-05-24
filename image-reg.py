import util
import numpy as np
import matplotlib.pyplot as plt

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
    template_data = util.load_template_data("Data/GG-366-FLAIR-1.0mm.nii")
    pat1_data = util.load_patient_scans("Data/Original/101_8_30_15")
    pat2_data = util.load_patient_scans("Data/Original/108_9_18_15")

    pat1_res = register_image(template_data, pat1_data)
    pat2_res = register_image(template_data, pat2_data)

    print("Template:")
    print(template_data.shape)
    util.multi_slice_viewer(template_data)

    # print("Patient 1:")
    # print(np.shape(pat1_data))
    # util.multi_slice_viewer(pat1_data)

    # print("Patient 2:")
    # print(np.shape(pat2_data))
    # util.multi_slice_viewer(pat2_data)

    print("Patient 1 Registration:")
    print(np.shape(pat1_res))
    util.multi_slice_subplot(pat1_res)

    print("Patient 2 Registration:")
    print(np.shape(pat2_res))
    util.multi_slice_subplot(pat2_res)

    plt.show()

    # sitk.WriteImage(resultImage, 'C:\\Users\\shb20\\Desktop\\resultImage.png')

if __name__ == '__main__':
    main()