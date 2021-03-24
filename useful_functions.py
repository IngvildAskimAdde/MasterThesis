
import os
import ImageViewer as iv
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

###### Folders and path functions ###########################
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


def get_array_from_image(path):
    """
    Reads an image from the given path, and returns a flatten image-array and image size

    :param path: path to image
    :return: flatten image array and image size
    """
    image = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(image)
    image_size = np.shape(array)
    array = array.flatten()
    return array, image_size


###### Visualization of images ##############################
def show_image_interactive(image_path, mask_path, view_mode):

    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)

    v = iv.Viewer(view_mode=view_mode, mask_to_show=['a'])
    v.set_image(image, label='image')
    v.set_mask(mask, label='mask')
    v.show()

def plot_2x2(image1_path, image2_path, image3_path, image4_path, slice_number):

    image1 = sitk.ReadImage(image1_path)
    image2 = sitk.ReadImage(image2_path)
    image3 = sitk.ReadImage(image3_path)
    image4 = sitk.ReadImage(image4_path)

    image1_array = sitk.GetArrayFromImage(image1)
    image2_array = sitk.GetArrayFromImage(image2)
    image3_array = sitk.GetArrayFromImage(image3)
    image4_array = sitk.GetArrayFromImage(image4)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(8, 3),
                                        sharex=True, sharey=True)
    for aa in (ax1, ax2, ax3, ax4):
        aa.set_axis_off()

    ax1.imshow(image1_array[slice_number], cmap='gray')
    ax1.set_title('Source')
    ax2.imshow(image2_array[slice_number], cmap='gray')
    ax2.set_title('Reference')
    ax3.imshow(image3_array[slice_number], cmap='gray')
    ax3.set_title('Matched')
    ax4.imshow(image4_array[slice_number], cmap='gray')
    ax4.set_title('Matched')

    plt.tight_layout()
    plt.show()


im1_Oxy = '/Volumes/LaCie/MasterThesis_Ingvild/Oxy_cropped/Oxytarget_90_PRE/T2.nii'
im1_LARC = '/Volumes/LaCie/MasterThesis_Ingvild/LARC_cropped/LARC-RRP-035/image.nii'
im2_Oxy = '/Volumes/LaCie/MasterThesis_Ingvild/Oxy_cropped_ZScoreNorm/Oxytarget_90_PRE/T2.nii'
im2_LARC = '/Volumes/LaCie/MasterThesis_Ingvild/LARC_cropped_ZScoreNorm/LARC-RRP-035/image.nii'

plot_2x2(im1_Oxy, im1_LARC, im2_Oxy, im2_LARC, 20)