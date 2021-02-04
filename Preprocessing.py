
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


#LARC_patientPaths, LARC_PatientNames, LARC_imagePaths, LARC_maskPaths = gd.get_paths('/Volumes/Untitled 1/LARC_T2_cleaned_nii', 'image', 'label.nii')
#Oxy_patientPaths, Oxy_PatientNames, Oxy_imagePaths, Oxy_maskPaths = gd.get_paths('/Volumes/Untitled/Ingvild_Oxytarget', 'T2', 'an.nii')

#LARC_df = dataframe(LARC_patientPaths, LARC_PatientNames, LARC_imagePaths, LARC_maskPaths)
#LARC_df = dimensions(LARC_df)
#crop_images(LARC_df, 512, 352)


"""
#image_original, mask_original, image_cropped, mask_cropped =
#crop_images('/Volumes/Untitled/LARC_T2_cleaned_nii/LARC-RRP-005/MRS1/image.nii',
                                                                         '/Volumes/Untitled/LARC_T2_cleaned_nii/LARC-RRP-005/MRS1/1 RTSTRUCT LARC_MRS1-label.nii',
                                                                         512,
                                                                     352)
"""

#image_original = sitk.ReadImage('//Volumes/HARDDISK/MasterThesis/Oxy_cropped/Oxytarget_128_PRE/T2.nii')
#mask_original = sitk.ReadImage('/Volumes/HARDDISK/MasterThesis/Oxy_cropped/Oxytarget_128_PRE/Manual_an.nii')

#print(image_original.GetSize())

#v = iv.Viewer()
#v.set_image(image_original, label='original image')
#v.set_mask(mask_original, label='original mask', color_name='green')
#v.set_image(image_cropped, label='cropped image')
#v.set_mask(mask_cropped, label='cropped mask',color_name='red')
#v.show()


#df = dataset_dataframe('/Volumes/Untitled/LARC_T2_cleaned_nii')
#df = dataset_dataframe('/Volumes/Untitled 1/Oxytarget_preprocessed')

#df = dimensions(df)
#df_imgFile = df.iloc[:int(df.shape[0]/2)]
#df_maskFile = df.iloc[2*int(df.shape[0]/3):]

#print('Image files: ', df_imgFile['xDimension'].value_counts())
#print('Mask files: ', df_maskFile['xDimension'].value_counts())
#print('Image files (voxelsize): ', df_imgFile['xVoxelDimension'].value_counts())
"""
print('Max x-dimension:', df['xDimension'].max())
print('Max y-dimension:', df['yDimension'].max())
print('Max z-dimension:', df['zDimension'].max())


plt.figure()
sns.histplot(df_imgFile, x='xDimension')
plt.title('OxyTarget image dimensions \n (512,512)=107, (528,528)=2, (560,560)=1')

plt.figure()
sns.histplot(df_imgFile, x='xVoxelDimension')
plt.title('OxyTarget voxelsizes')
          #'\n (0.323,0.323)=1, (0.340,0.340)=1, (0.342,0.342)=1, (0.344, 0.344)=1 \n (0.3467,0.3467)=1, (0.3477,0.3477)=1, (0.3478,0.3478)=1, (0.351,0.351)=1 \n (0.352,0.352)=65 ')
"""