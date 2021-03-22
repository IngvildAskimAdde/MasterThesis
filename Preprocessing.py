
import os
import pandas as pd
import SimpleITK as sitk
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ImageViewer as iv
import get_data as gd


def dataframe(patientPaths, patientNames, imagePaths, maskPaths):

    df = pd.DataFrame(list(zip(patientPaths, patientNames, imagePaths, maskPaths)),columns=['patientPaths', 'ID', 'imagePaths', 'maskPaths'])
    return df


def dimensions(dataset_dataframe):

    dataset_dataframe['xDimension'] = ''
    dataset_dataframe['yDimension'] = ''
    dataset_dataframe['zDimension'] = ''

    dataset_dataframe['xVoxelDimension'] = ''
    dataset_dataframe['yVoxelDimension'] = ''
    dataset_dataframe['zVoxelDimension'] = ''

    for i in range(len(dataset_dataframe['imagePaths'])):
        image = sitk.ReadImage(dataset_dataframe['imagePaths'][i])
        array = sitk.GetArrayFromImage(image)
        dim = np.shape(array)

        dataset_dataframe['xDimension'][i] = dim[1]
        dataset_dataframe['yDimension'][i] = dim[2]
        dataset_dataframe['zDimension'][i] = dim[0]

        dataset_dataframe['xVoxelDimension'][i] = image.GetSpacing()[0]
        dataset_dataframe['yVoxelDimension'][i] = image.GetSpacing()[1]
        dataset_dataframe['zVoxelDimension'][i] = image.GetSpacing()[2]

        print('Dimension calculated for patient:', dataset_dataframe['imagePaths'][i])

    return dataset_dataframe


def get_array(path):

    """
    Input: Path to image files
    Output: sitk array of image file, and image size
    """

    img = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(img)
    imsize = np.shape(array)
    #array = array.flatten()
    return array, imsize


def create_image_from_array(array, imsize):
    """
    Input: Image as array, and image size
    Output: Image object
    """
    im = np.reshape(array,imsize)
    im = im.astype(int)
    im = sitk.GetImageFromArray(im)
    return im

