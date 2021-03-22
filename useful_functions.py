
import os
import ImageViewer as iv
import SimpleITK as sitk

def create_folder(src_main_folder, dst_main_folder, patient_identifier):
    """
    Input: Path to source folder, path to destination folder, patient
    identifier (LARC-RRP or ID)

    Output: New folder with the same structure as source folder
    """

    patient_list = [i for i in os.listdir(src_main_folder) if i.startswith(patient_identifier)]

    if patient_identifier == 'LARC-RRP':
        scan_time_point_folder = 'MRS1'
        for patient in patient_list:
            os.makedirs(os.path.join(dst_main_folder, patient, scan_time_point_folder))

    elif patient_identifier == 'Oxytarget':
        for patient in patient_list:
            os.makedirs(os.path.join(dst_main_folder, patient))

    elif patient_identifier == 'ID':
        for patient in patient_list:
            os.makedirs(os.path.join(dst_main_folder, patient))

    else:
        print('No valid patient identifier')

def create_dst_paths(dst_main_folder):
    """
    Creates a list of destination paths based on the structure of the dst_main_folder path

    :param dst_main_folder: path to main folder containing subfolders for each patient
    :return: list of destination paths for each patient
    """
    dst_subfolder_list = [f for f in os.listdir(dst_main_folder)
                          if os.path.isdir(os.path.join(dst_main_folder, f))]

    dst_list = []

    for i in range(len(dst_subfolder_list)):
        dst_list.append(dst_main_folder + '/' + dst_subfolder_list[i])

    return dst_list


def show_image_interactive(image_path, mask_path, view_mode):

    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)

    v = iv.Viewer(view_mode=view_mode, mask_to_show=['a'])
    v.set_image(image, label='image')
    v.set_mask(mask, label='mask')
    v.show()