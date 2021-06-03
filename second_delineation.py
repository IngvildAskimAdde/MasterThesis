
import get_data as gd
import Preprocessing as p
import SimpleITK as sitk
import useful_functions as uf
import crop_images
import h5py
import numpy as np
import pandas as pd
import CreateHD5F

def correct_patients_csv(csv_path1, csv_path2):
    """
    Returns dataframes with patients which has the second delineation as well as the first

    :param csv_path1: path to the csv file with dsc scores of the first mask delineation
    :param csv_path2: path to the csv file with dsc scores of the second mask delineation
    :return: two dataframes with the correct patients
    """
    unwanted_patient_ids = [1028, 1097, 1106, 1108, 1111, 1130, 1162, 1169, 1188, 1192, 1190, 1174, 1069, 1074, 1094]

    df1 = pd.read_csv(csv_path1)
    df2 = pd.read_csv(csv_path2)

    assert df1.shape[0] == df2.shape[0], "Shape does not match"
    row = []
    for i in range(df1.shape[0]):
        id = df1['patient_ids'].values[i]
        if id in unwanted_patient_ids:
            row.append(i)

    df1 = df1.drop(row)
    df2 = df2.drop(row)
    return df1, df2


def interobserver_variations_on_val(main_folder_path, df1, df2):
    """
    Calculates the dsc scores between the two delineations in the validation set of the OxyTarget data

    :param main_folder_path: path to folder with patients which has two delineations
    :return: list of dsc scores for each patient and median value
    """
    val_patients = ['OxyTarget_115_PRE', 'OxyTarget_121_PRE', 'OxyTarget_122_PRE', 'OxyTarget_124_PRE', 'OxyTarget_133_PRE', 'OxyTarget_138_PRE', 'OxyTarget_163_PRE', 'OxyTarget_164_PRE', 'OxyTarget_184_PRE', 'OxyTarget_43_PRE',  'OxyTarget_61_PRE']
    dsc_scores = []

    for i in val_patients:
        paths_mask1 = main_folder_path + '/' + i + '/Manual_an.nii'
        paths_mask2 = main_folder_path + '/' + i + '/Manual_shh.nii'
        print(paths_mask1)
        print(paths_mask2)
        mask1 = sitk.ReadImage(paths_mask1)
        mask2 = sitk.ReadImage(paths_mask2)

        dsc = uf.calculate_dice(mask1, mask2)
        print(paths_mask1, dsc)
        dsc_scores.append(dsc)

    df3 = pd.DataFrame(dsc_scores, columns=['f1_score'])

    print(np.median(list(df1['f1_score'])))
    print(np.median(list(df2['f1_score'])))
    print(np.median(list(df3['f1_score'])))

    df = pd.DataFrame()
    df[r'Radiologist$_{\mathrm{O}}^{\mathrm{1}}}$'] = df1['f1_score']
    df[r'Radiologist$_{\mathrm{O}}^{\mathrm{2}}}$'] = df2['f1_score']
    df['Interobserver'] = list(df3['f1_score'])

    uf.scatter_plot_masks(df1['f1_score'], df2['f1_score'], df3['f1_score'], df1['patient_ids'], color='#9ecae1', markers=['^', 'v', '*'])

    #df = pd.concat([df1, df2, df3])
    #df = df.drop(['patient_ids'], axis=1)
    #df = pd.melt(df, id_vars=['Mask'], var_name=['Parameters'])

    return df

def interobserver_variations_allPatients(main_folder_path):
    """
    Calculates the dsc scores between the two delineation masks in all of the patients in the OxyTarget data

    :param main_folder_path: path to folder with patients which has two delineations
    :return: list of dsc scores for each patient
    """

    Oxy_patientPaths1, Oxy_PatientNames1, Oxy_imagePaths1, Oxy_maskPaths1 = gd.get_paths(main_folder_path, 'T2', 'an.nii')
    Oxy_patientPaths2, Oxy_PatientNames2, Oxy_imagePaths2, Oxy_maskPaths2 = gd.get_paths(main_folder_path, 'T2', 'shh.nii')

    Oxy_maskPaths1_new = []
    Oxy_maskPaths2_new = []
    for i in range(len(Oxy_maskPaths1)):
        if Oxy_maskPaths1[i].endswith('an.nii'):
            Oxy_maskPaths1_new.append(Oxy_maskPaths1[i])
        if Oxy_maskPaths2[i].endswith('shh.nii'):
            Oxy_maskPaths2_new.append(Oxy_maskPaths2[i])

    assert len(Oxy_maskPaths1_new) == len(Oxy_maskPaths2_new), "Shapes do not match"
    dsc_scores = []
    for i in range(len(Oxy_maskPaths1_new)):
        mask1 = sitk.ReadImage(Oxy_maskPaths1_new[i])
        mask2 = sitk.ReadImage(Oxy_maskPaths2_new[i])
        dsc = uf.calculate_dice(mask1, mask2)
        dsc_scores.append(dsc)

    print(np.median(dsc_scores))

    df = pd.DataFrame([Oxy_maskPaths1_new, Oxy_maskPaths2_new, dsc_scores]).T

    return df

if __name__ == '__main__':

    df1, df2 = correct_patients_csv('/Volumes/LaCie/MasterThesis_Ingvild/Experiments/Oxy_new/Oxy_ID_26_new/patient.csv', '/Volumes/LaCie/MasterThesis_Ingvild/Experiments/Oxy_new/Oxy_ID_26_new/mask2/patient.csv')
    interobserver_dsc = interobserver_variations_allPatients('/Volumes/LaCie/MasterThesis_Ingvild/Data/Oxy/Oxy_secondDelineationPatients_cropped')
    print(min(interobserver_dsc))
    df = interobserver_variations_on_val('/Volumes/LaCie/MasterThesis_Ingvild/Data/Oxy/Oxy_secondDelineationPatients_cropped', df1, df2)
    #colors_Oxy = ['#9ecae1']
    #colnames = list(df.columns)

    #uf.violinplot_version2(df, 20, 20, '', colnames, colors_Oxy)

    #uf.show_image_interactive('/Volumes/LaCie/MasterThesis_Ingvild/Data/Oxy_secondDelineationPatients_cropped/Oxytarget_110_PRE/T2.nii', '/Volumes/LaCie/MasterThesis_Ingvild/Data/Oxy_secondDelineationPatients_cropped/Oxytarget_110_PRE/Manual_an.nii', '2')
    #uf.show_image_interactive('/Volumes/LaCie/MasterThesis_Ingvild/Data/Oxy_secondDelineationPatients_cropped/Oxytarget_110_PRE/T2.nii', '/Volumes/LaCie/MasterThesis_Ingvild/Data/Oxy_secondDelineationPatients_cropped/Oxytarget_110_PRE/Manual_shh.nii', '2')
