
import get_data as gd
import Preprocessing as p
import SimpleITK as sitk
import useful_functions as uf
import crop_images
import h5py
import numpy as np


def performance_mask_ssh(prediction_path, mask_paths, threshold):

    with h5py.File(prediction_path, 'r') as f:
        y_pred = f['predicted']
        slices = y_pred.shape[0]

        dsc_scores = []

        for i in range(slices):
            print(i)
            predicted = f['predicted'][i]
            predicted = (predicted > threshold).astype(predicted.dtype)
            predicted = predicted.flatten()
            #idx = f['patient_ids'][i]
            #print(idx)








if __name__ == '__main__':
    """
    #Get paths
    Oxy_patientPaths_1, Oxy_PatientNames_1, Oxy_imagePaths_1, Oxy_maskPaths_1_initial = gd.get_paths('/Volumes/LaCie/MasterThesis_Ingvild/Data/Oxy_secondDelineationPatients_cropped', 'T2', 'an.nii')
    Oxy_patientPaths_2, Oxy_PatientNames_2, Oxy_imagePaths_2, Oxy_maskPaths_2_initial = gd.get_paths('/Volumes/LaCie/MasterThesis_Ingvild/Data/Oxy_secondDelineationPatients_cropped', 'T2', 'ssh.nii')

    #Create list of mask paths with patients which have the second delineation file 'Manual_shh.nii'
    Oxy_maskPaths_2 = []
    for i in range(len(Oxy_maskPaths_2_initial)):
        if Oxy_maskPaths_2_initial[i].endswith('shh.nii'):
            Oxy_maskPaths_2.append(Oxy_maskPaths_2_initial[i])

    Oxy_maskPaths_1 = []
    for j in range(len(Oxy_maskPaths_1_initial)):
        if Oxy_maskPaths_1_initial[j].endswith('an.nii'):
            Oxy_maskPaths_1.append(Oxy_maskPaths_1_initial[j])

    #performance_mask_ssh('/Volumes/LaCie/MasterThesis_Ingvild/Experiments/Oxy_new/Oxy_ID_24_new/prediction.092.h5', Oxy_maskPaths_2, 0.5)
    """

    uf.show_image_interactive('/Volumes/LaCie/MasterThesis_Ingvild/Data/Oxy_secondDelineationPatients_cropped/Oxytarget_110_PRE/T2.nii', '/Volumes/LaCie/MasterThesis_Ingvild/Data/Oxy_secondDelineationPatients_cropped/Oxytarget_110_PRE/Manual_an.nii', '2')
    uf.show_image_interactive('/Volumes/LaCie/MasterThesis_Ingvild/Data/Oxy_secondDelineationPatients_cropped/Oxytarget_110_PRE/T2.nii', '/Volumes/LaCie/MasterThesis_Ingvild/Data/Oxy_secondDelineationPatients_cropped/Oxytarget_110_PRE/Manual_shh.nii', '2')
