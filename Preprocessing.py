

import pandas as pd
import SimpleITK as sitk
import numpy as np
import get_data as gd



def dataframe(patientPaths, patientNames, imagePaths, maskPaths):
    """
    Creates a dataframe containing paths to patient folder, to image, to mask and name of patient

    :param patientPaths: list of paths to patient folders
    :param patientNames: list of patient names
    :param imagePaths: list of paths to image-files
    :param maskPaths: list of paths to mask-files
    :return: dataframe with information
    """
    df = pd.DataFrame(list(zip(patientPaths, patientNames, imagePaths, maskPaths)),columns=['patientPaths', 'ID', 'imagePaths', 'maskPaths'])
    return df


def dimensions(dataset_dataframe):
    """
    Adds information about image dimensions to dataframe

    :param dataset_dataframe: dataframe with information
    :return: dataframe with dimension information added
    """
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

def create_dataframe(folder_path, image_prefix, mask_suffix):
    """
    Returns a dataframe with information of the images in the folder_path

    folder_path: path to folder with images
    image_prefix: prefix of image files
    mask_suffix: suffix of mask files
    """
    patientPaths, patientNames, imagePaths, maskPaths = gd.get_paths(folder_path, image_prefix, mask_suffix)
    df = dataframe(patientPaths, patientNames, imagePaths, maskPaths)
    df = dimensions(df)
    return df


