
import Preprocessing as p
import get_data as gd
import crop_images as crop
import pandas as pd

def dwi_path(patientPaths):
    """
    Creates dictionary with paths to dwi, t2 and mask
    :param patientPaths: list of paths to patient folder with the images
    :return: dictionary with paths
    """
    mask_paths = []
    t2_paths = []
    dwi = {'b0':[], 'b1':[], 'b2':[], 'b3':[], 'b4':[], 'b5':[], 'b6':[], }

    for patient in patientPaths:
        for b in dwi.keys():
            path = patient + '/img_DW_' + b + '.nii'
            dwi[b].append(path)

        mask_paths.append(patient + '/Manual_an.nii')
        t2_paths.append(patient + '/T2.nii')

    dwi['t2'] = t2_paths
    dwi['mask'] = mask_paths


    return dwi


if __name__ == '__main__':

    Oxy_patientPaths, Oxy_patientNames, Oxy_imagePaths, Oxy_maskPaths = gd.get_paths('/Volumes/LaCie/MasterThesis_Ingvild/Data/Oxy/TumorSlices/Oxy_cropped_TS', 'T2', 'an.nii')
    dwiPaths = dwi_path(Oxy_patientPaths)
    df = pd.DataFrame(dwiPaths)

    #df = pd.DataFrame([dwiPaths, maskPaths], columns={'dwiPaths', 'maskPaths'})
    #df = p.dimensions(df)
    crop.crop_t2_dwi_mask(df, 352)