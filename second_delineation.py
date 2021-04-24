
import get_data as gd

Oxy_patientPaths, Oxy_PatientNames, Oxy_imagePaths, Oxy_maskPaths = gd.get_paths('/Volumes/LaCie/MasterThesis_Ingvild/Data/Oxy_cropped_MatchedHistZScore', 'T2', 'an.nii')
Oxy_patientPaths_2, Oxy_PatientNames_2, Oxy_imagePaths_2, Oxy_maskPaths_2_initial = gd.get_paths('/Volumes/LaCie/MasterThesis_Ingvild/Data/DWandOtherContour_updatedPatients', 'T2', 'ssh.nii')
Oxy_maskPaths.remove('/Volumes/LaCie/MasterThesis_Ingvild/Data/Oxy_cropped_MatchedHistZScore/Oxytarget_172_PRE/Manual_an.nii')
Oxy_maskPaths.remove('/Volumes/LaCie/MasterThesis_Ingvild/Data/Oxy_cropped_MatchedHistZScore/Oxytarget_190_PRE/Manual_an.nii')

Oxy_maskPaths_2 = []
for i in range(len(Oxy_maskPaths_2_initial)):
    if Oxy_maskPaths_2_initial[i].endswith('shh.nii'):
        Oxy_maskPaths_2.append(Oxy_maskPaths_2_initial[i])


