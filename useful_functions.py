
import os
import ImageViewer as iv
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from matplotlib.lines import Line2D
import pandas as pd

matplotlib.rcParams.update({'font.size': 25})
matplotlib.rcParams['font.family'] = "serif"
matplotlib.rcParams.update({'xtick.labelsize': 20})

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


####### Get data functions ##################################
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

def dataframe_of_f1scores(excel_path: str, sheet_name: str, filetype: list, range_ids, patient_ids=False):
    """
    The function takes in an excel file and creates a list of paths to the wanted file(s) (filetype).
    The wanted files are then converted to dataframes, and a singel dataframe containing all the
    'f1_scores' in the wanted files is returned.


    excel_path: path to excel file containing the paths to folders with experiment results
    sheet_name: name of the excel sheet you want to create a dataframe of
    filetype: filename of the file you want to collect from the result folder

    Returns a dataframe with all of f1_scores for the filetype for a given dataset.
    """
    dataframe = pd.read_excel(excel_path, sheet_name=sheet_name)  # Create dataframe of excel sheet
    paths = list(dataframe['Result path'])  # Get a list of folderpaths were the results are saved
    learning_rates = list(dataframe['Learning rate'])
    loss_functions = list(dataframe['Loss function'])
    IDs = list(dataframe['Experiment ID'])
    file_paths_352 = []  # List of paths for images with dimension 352x352
    file_paths_256 = []  # List of paths for images with dimension 256x256
    df = pd.DataFrame()

    for file in filetype:

        for i in range_ids:

            if (dataframe['Dataset'][
                0] == 'Oxytarget'):  # If the dataset is Oxytarget then all the images have the same dimensions.
                paths[i] = paths[i] + '/' + file  # List of paths for the patient.csv files
                if os.path.isfile(paths[i]):  # Check if the file path exists
                    if df.empty:  # Check if the dataframe is empty
                        df = pd.read_csv(paths[i])  # Add result to dataframe
                        df.rename(columns={'f1_score': '{:.0e}'.format(learning_rates[i]) + '+' + loss_functions[i]},
                                  inplace=True)  # Rename column
                        # df = df.rename(columns={'f1_score': IDs[i]})
                    else:
                        # If the dataframe is not empty, add columns with the score from the other results
                        df_temp = pd.read_csv(paths[i])
                        df = df.join(df_temp.set_index('patient_ids'), on='patient_ids')
                        df.rename(columns={'f1_score': '{:.0e}'.format(learning_rates[i]) + '+' + loss_functions[i]},
                                  inplace=True)
                        # df = df.rename(columns={'f1_score': IDs[i]})



            else:  # For other datasets the images are of two different dimensions
                if file.endswith('352.csv'):
                    file_paths_352.append(str(paths[i]) + '/' + file)
                else:
                    file_paths_256.append(str(paths[i]) + '/' + file)

    if not dataframe['Dataset'][0] == 'Oxytarget':
        for i in range(len(file_paths_352)):
            if os.path.isfile(file_paths_352[i]) and os.path.isfile(file_paths_256[i]):

                if df.empty:  # Check if the dataframe is empty
                    df_352 = pd.read_csv(file_paths_352[i])
                    df_256 = pd.read_csv(file_paths_256[i])
                    df = df_352.append(df_256, ignore_index=True)
                    df.rename(columns={
                        'f1_score': '{:.0e}'.format(learning_rates[range_ids[i]]) + '+' + loss_functions[range_ids[i]]},
                              inplace=True)  # Rename column
                    # df = df.rename(columns={'f1_score': IDs[i]})

                else:
                    # If the dataframe is not empty, add columns with the score from the other results
                    df_352 = pd.read_csv(file_paths_352[i])
                    df_256 = pd.read_csv(file_paths_256[i])
                    df_temp = df_352.append(df_256, ignore_index=True)
                    df = df.join(df_temp.set_index('patient_ids'), on='patient_ids')
                    df.rename(columns={
                        'f1_score': '{:.0e}'.format(learning_rates[range_ids[i]]) + '+' + loss_functions[range_ids[i]]},
                              inplace=True)  # Rename column
                    # df = df.rename(columns={'f1_score': IDs[i]})

    if not patient_ids:
        df = df.drop(['patient_ids'], axis=1)

    return df


def swap_columns(df, c1, c2):
    """
    The function swaps the position of two columns (c1 and c2) in a dataframe (df).
    Returns new dataframe with swapped columns.
    """
    df['temp'] = df[c1]
    df[c1] = df[c2]
    df[c2] = df['temp']
    df.drop(columns=['temp'], inplace=True)
    df = df.rename(columns={c1:c2, c2:c1})
    return df


###### Visualization of images ##############################
def show_image_interactive(image_path, mask_path, view_mode):

    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)

    v = iv.Viewer(view_mode=view_mode, mask_to_show=['a'])
    v.set_image(image, label='image')
    v.set_mask(mask, label='mask')
    v.show()

def show_image(image_path, mask_path, slice_number):

    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)
    mask = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayFromImage(mask)

    plt.figure(figsize=(11,8))
    plt.imshow(image_array[slice_number], cmap='gray')#, vmin=-2, vmax=4)
    plt.imshow(mask_array[slice_number], cmap='gray', alpha=0.5)
    plt.axis('off')
    plt.show()


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


#im1_Oxy = '/Volumes/LaCie/MasterThesis_Ingvild/Oxy_cropped/Oxytarget_90_PRE/T2.nii'
#im1_LARC = '/Volumes/LaCie/MasterThesis_Ingvild/LARC_cropped/LARC-RRP-035/image.nii'
#im2_Oxy = '/Volumes/LaCie/MasterThesis_Ingvild/Oxy_cropped_ZScoreNorm/Oxytarget_90_PRE/T2.nii'
#im2_LARC = '/Volumes/LaCie/MasterThesis_Ingvild/LARC_cropped_ZScoreNorm/LARC-RRP-035/image.nii'

#plot_2x2(im1_Oxy, im1_LARC, im2_Oxy, im2_LARC, 20)

def plot_pixel_distribution(df):
    """
    Plots the pixel distribution of the images with paths given by a dataframe

    :param df: dataframe with image paths
    :return: a plot of the distribution of pixel intensities in the images given by the dataframe
    """
    plt.figure(figsize=(14, 8))

    for i in range(len(df['imagePaths'])):
        print('Plotting:', df['imagePaths'][i])
        array, image_size = get_array_from_image(df['imagePaths'][i])

        if df['ID'][i].endswith('PRE'):
            sns.kdeplot(data=array.flatten(), color='#9ecae1')
        else:
            sns.kdeplot(data=array.flatten(), color='#fdae6b')

    legend_labels = [Line2D([0], [0], color='#9ecae1', label='OxyTarget'),
                     Line2D([0], [0], color='#fdae6b', label='LARC-RRP')]

    plt.xlabel('Pixel intensity')
    plt.legend(handles=legend_labels)
    plt.show()

    #plt.savefig('savfig.pdf')

def plot_matched_images(source_image_path, reference_image_path, matched_image_path, slice_number, slice_number_ref):

    source_image = sitk.ReadImage(source_image_path)
    source_image_array = sitk.GetArrayFromImage(source_image)
    ref_image = sitk.ReadImage(reference_image_path)
    ref_image_array = sitk.GetArrayFromImage(ref_image)
    matched_image = sitk.ReadImage(matched_image_path)
    matched_image_array = sitk.GetArrayFromImage(matched_image)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(14, 5),
                                        sharex=True, sharey=True)

    for aa in (ax1, ax2, ax3):
        aa.set_axis_off()

    ax1.imshow(source_image_array[slice_number], cmap='gray')
    #ax1.set_title('Input Image')
    ax2.imshow(ref_image_array[slice_number_ref], cmap='gray')
    #ax2.set_title('Reference Image')
    ax3.imshow(matched_image_array[slice_number], cmap='gray')
    #ax3.set_title('Matched Image')

    plt.tight_layout()
    plt.show()


def violinplot(dataframe, fontsize, labelsize, title, colors):

    plt.figure(figsize=(11, 8))

    matplotlib.rcParams.update({'font.size': fontsize})
    matplotlib.rcParams['font.family'] = "serif"
    matplotlib.rcParams.update({'xtick.labelsize': labelsize})
    sns.violinplot(x=dataframe['Parameters'], y=dataframe['value'], palette=colors)
    plt.xlabel(None)
    plt.ylabel('DSC')
    plt.title(title)
    plt.xticks(rotation=30)
    plt.ylim(-0.35, 1.15)
    plt.tight_layout()
    plt.show()

def boxplot(dataframe, fontsize, labelsize, title, colors):

    plt.figure(figsize=(11, 8))

    matplotlib.rcParams.update({'font.size': fontsize})
    matplotlib.rcParams['font.family'] = "serif"
    matplotlib.rcParams.update({'xtick.labelsize': labelsize})
    sns.boxplot(x=dataframe['Parameters'], y=dataframe['value'], palette=colors)
    plt.xlabel(None)
    plt.ylabel('DSC')
    plt.title(title)
    plt.xticks(rotation=30)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.show()

