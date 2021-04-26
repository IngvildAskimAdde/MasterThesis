
import get_data as gd
import Preprocessing as p
import SimpleITK as sitk
import rescale_masks



if __name__ == '__main__':

    #Get paths
    Oxy_patientPaths, Oxy_PatientNames, Oxy_imagePaths, Oxy_maskPaths = gd.get_paths('/Volumes/LaCie/MasterThesis_Ingvild/Data/Oxy_cropped_MatchedHistZScore', 'T2', 'an.nii')
    Oxy_patientPaths_2, Oxy_PatientNames_2, Oxy_imagePaths_2, Oxy_maskPaths_2_initial = gd.get_paths('/Volumes/LaCie/MasterThesis_Ingvild/Data/Oxy_secondDelineationPatients', 'T2', 'ssh.nii')

    #Create list of mask paths with patients which have the second delineation file 'Manual_shh.nii'
    Oxy_maskPaths_2 = []
    for i in range(len(Oxy_maskPaths_2_initial)):
        if Oxy_maskPaths_2_initial[i].endswith('shh.nii'):
            Oxy_maskPaths_2.append(Oxy_maskPaths_2_initial[i])

    #Create dataframe
    Oxy_df = p.dataframe(Oxy_patientPaths_2, Oxy_PatientNames_2, Oxy_imagePaths_2, Oxy_maskPaths_2)
    Oxy_df = p.dimensions(Oxy_df)

    for i in range(len(Oxy_df['maskPaths'])):
        print(Oxy_df['maskPaths'][i])
        mask = sitk.ReadImage(Oxy_df['maskPaths'][i])
        print(mask.GetSize())
        print(sitk.GetArrayFromImage(mask).max())


