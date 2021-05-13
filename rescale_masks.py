
import SimpleITK as sitk
import numpy as np
import os
import get_data as gd
import Preprocessing as p

def rescale(dataframe, tumor_value):
    """
    Rescales the masks to 0 and 1, and saves the masks.

    :param dataframe: dataframe with mask paths
    :param tumor_value: original value of tumor voxels
    :return: saves the rescaled masks to the same folder as previously stored in
    """

    for i in range(len(dataframe['maskPaths'])):
        print(dataframe['maskPaths'][i])
        mask = sitk.ReadImage(dataframe['maskPaths'][i])
        mask_array = sitk.GetArrayFromImage(mask)
        imsize = np.shape(mask_array)
        mask_flatten = mask_array.flatten()
        print(np.unique(mask_flatten))

        for pixel in range(len(mask_flatten)):
            mask_flatten[pixel] = int(mask_flatten[pixel] / tumor_value)


        mask_rescaled = np.reshape(mask_flatten, imsize)
        #mask_rescaled = mask_rescaled.astype(int)
        mask_rescaled = sitk.GetImageFromArray(mask_rescaled)

        os.remove(dataframe['maskPaths'][i])
        sitk.WriteImage(mask_rescaled, dataframe['maskPaths'][i])

        print(np.unique(mask_rescaled))


Oxy_patientPaths, Oxy_PatientNames, Oxy_imagePaths, Oxy_maskPaths = gd.get_paths('/Volumes/LaCie/MasterThesis_Ingvild/Data/dwi/Oxy_all', 'T2', 'an.nii')
Oxy_df = p.dataframe(Oxy_patientPaths, Oxy_PatientNames, Oxy_imagePaths, Oxy_maskPaths)

#rescale(Oxy_df, 1000)

