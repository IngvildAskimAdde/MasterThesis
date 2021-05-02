
import os
import ImageViewer as iv
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from matplotlib.lines import Line2D
import pandas as pd
import h5py
from mpl_toolkits.axes_grid1 import make_axes_locatable

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


####### Performance ########################################
def calculate_dice(mask_a, mask_b):
    """
    Calculate DICE score for two binary masks (=sitk images)
    """
    npa1 = sitk.GetArrayFromImage(mask_a)
    npa2 = sitk.GetArrayFromImage(mask_b)

    dice = 2*np.count_nonzero(npa1 & npa2) / (np.count_nonzero(npa1) + np.count_nonzero(npa2))
    return dice


def max_and_min_dsc_score(df):

    #Create dataframe of slices
    df.drop(df.index[df['f1_score'] == 1.0], inplace=True)

    #Find maximum dsc score
    max_score = df['f1_score'].max()
    print(df[df['f1_score'] == max_score].index.values)
    print(max_score)

    # Find minimum dsc score
    min_score = df['f1_score'].min()
    print(df[df['f1_score'] == min_score].index.values)
    print(min_score)

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

    im1 = ax1.imshow(source_image_array[slice_number], cmap='gray')
    plt.colorbar(im1, ax=ax1, shrink=0.67)
    #ax1.set_title('Input Image')
    im2 = ax2.imshow(ref_image_array[slice_number_ref], cmap='gray')
    plt.colorbar(im2, ax=ax2, shrink=0.67)
    #ax2.set_title('Reference Image')
    im3 = ax3.imshow(matched_image_array[slice_number], cmap='gray')
    plt.colorbar(im3, ax=ax3, shrink=0.67)
    #ax3.set_title('Matched Image')
    plt.tight_layout()
    plt.show()

def plot_slice_nifti(path1, slice):

    image = sitk.ReadImage(path1)
    image_array = sitk.GetArrayFromImage(image)

    plt.figure(figsize=(11,8))
    plt.imshow(image_array[slice], cmap='gray')
    plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def violinplot(dataframe, fontsize, labelsize, title, colors):

    plt.figure(figsize=(11, 8))

    matplotlib.rcParams.update({'font.size': fontsize})
    matplotlib.rcParams['font.family'] = "serif"
    matplotlib.rcParams.update({'xtick.labelsize': labelsize})
    #sns.violinplot(x=dataframe['Parameters'], y=dataframe['value'], palette=colors, scale='width', cut=0)
    sns.violinplot(x=dataframe['Mask'], y=dataframe['value'], palette=colors, scale='width', cut=0)
    plt.xlabel(None)
    plt.ylabel(r'DSC$_{\mathrm{P}}$')
    plt.title(title)

    #legend_elements = [Line2D([0], [0], color=colors[0], label='OxyTarget validation patients', markersize=15, lw=8)]
    #plt.legend(handles=legend_elements)

    #plt.xticks(rotation=30)
    #plt.ylim(-0.35, 1.15)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.show()

def boxplot(dataframe, fontsize, labelsize, title, colors):

    plt.figure(figsize=(11, 8))

    matplotlib.rcParams.update({'font.size': fontsize})
    matplotlib.rcParams['font.family'] = "serif"
    matplotlib.rcParams.update({'xtick.labelsize': labelsize})
    sns.boxplot(x=dataframe['Parameters'], y=dataframe['value'], palette=colors)
    plt.xlabel(None)
    plt.ylabel(r'Mean DSC$_{\mathrm{S}}$')
    plt.title(title)
    plt.xticks(rotation=30)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.show()


def scatter_plot(max_dsc_list, color):
    """
    Creates a satter plot of maximum dsc scores.

    :param max_dsc_list: list of maximum dsc scores for different folds
    :param color: color of scatter dots
    :return: scatter plot
    """

    x_values = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']
    plt.figure(figsize=(11, 8))
    plt.scatter(x_values, max_dsc_list, color=color)
    plt.ylabel('DSC')
    plt.xticks(rotation=30)
    plt.ylim(-0.05, 1.05)
    plt.show()

def plot_learning_rates(dictionary, x_labels, color=None, markers=None):
    """
    Plots the median dsc score obtain with different learning rates for the various datasets

    :param dictionary: dictionary of datasets with corresponding median dsc scores for different learning rates
    :param x_labels: list of names for datasets on the x-axis
    :param color: list of colors corresponding to a dataset
    :param markers: list of marker options
    :return: scatter plot of median dsc scores
    """
    count = 0
    plt.figure(figsize=(11,8))
    for key in dictionary:
        plt.scatter(x_labels, dictionary[key], color=color, marker=markers[count], facecolors='none', s=200, linewidths=2)
        count += 1

    legend_elements = [Line2D([0], [0], marker='o', color='k', label='1e-03',
                              markerfacecolor='none', markersize=15, linestyle='none'),
                       Line2D([0], [0], marker='s', color='k', label='1e-04',
                              markerfacecolor='none', markersize=15, linestyle='none'),
                       Line2D([0], [0], marker='^', color='k', label='1e-05',
                              markerfacecolor='none', markersize=15, linestyle='none')]

    plt.ylabel(r'DSC$_{\mathrm{P}}$')
    plt.xlabel(' ')
    plt.ylim(-0.05, 1.05)
    plt.legend(handles=legend_elements)#, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.tight_layout()
    plt.show()

def plot_loss_functions(dictionary, x_labels, color=None, markers=None):
    """
    Plots the median dsc score obtain with different loss functions for the various datasets

    :param dictionary: dictionary of datasets with corresponding median dsc scores for different loss functions
    :param x_labels: list of names for datasets on the x-axis
    :param color: list of colors corresponding to a dataset
    :param markers: list of marker options
    :return: scatter plot of median dsc scores
    """
    count = 0
    plt.figure(figsize=(11,8))
    for key in dictionary:
        plt.scatter(x_labels, dictionary[key], color=color, marker=markers[count], facecolors='none', s=200, linewidths=2)
        count += 1

    legend_elements = [Line2D([0], [0], marker='X', color='k', label='Dice',
                              markerfacecolor='none', markersize=15, linestyle='none'),
                       Line2D([0], [0], marker='D', color='k', label='Modified Dice',
                              markerfacecolor='none', markersize=15, linestyle='none')]

    plt.ylabel(r'DSC$_{\mathrm{P}}$')
    plt.xlabel(' ')
    plt.ylim(-0.05, 1.05)
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.tight_layout()
    plt.show()

def scatter_plot_masks(dsc1, dsc2,  dsc3, patient_ids, color=None, markers=None):
    """
    Plots the median dsc score obtain with different masks for the various datasets

    :param dictionary: dictionary of datasets with corresponding median dsc scores for different loss functions
    :param x_labels: list of names for datasets on the x-axis
    :param color: list of colors corresponding to a dataset
    :param markers: list of marker options
    :return: scatter plot of median dsc scores
    """

    plt.figure(figsize=(11,8))
    plt.scatter(patient_ids, dsc1, color=color, marker=markers[0], facecolors='none', s=200, linewidths=2)
    plt.scatter(patient_ids, dsc2, color=color, marker=markers[1], facecolors='none', s=200, linewidths=2)
    plt.scatter(patient_ids, dsc3, color=color, marker=markers[2], facecolors='none', s=200, linewidths=2)


    legend_elements = [Line2D([0], [0], marker='^', color='k', label='Radiologist$_{\mathrm{O}}^{\mathrm{1}}}$',
                              markerfacecolor='none', markersize=15, linestyle='none'),
                       Line2D([0], [0], marker='v', color='k', label='Radiologist$_{\mathrm{O}}^{\mathrm{2}}}$',
                              markerfacecolor='none', markersize=15, linestyle='none'),
                       Line2D([0], [0], marker='*', color='k', label='Interobserver',
                              markerfacecolor='none', markersize=15, linestyle='none')
                       ]

    plt.ylabel(r'DSC$_{\mathrm{P}}$')
    plt.xlabel('Patient IDs')
    plt.ylim(-0.05, 1.05)
    plt.legend(handles=legend_elements)#, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.tight_layout()
    plt.show()

def plot_image_slice(prediction_path, indice, second_mask_path=None):
    """
    Function to plot an image slice with the corresponding predicted mask, ground truth mask (Mask 1)
    and optional second ground truth mask (Mask 2)

    :param prediction_path: path to predicted hdf5 file
    :param indice: slice number
    :param second_mask_path: path to hdf5 file with second delineations as target_an
    :return: plot of slice with corresponding delineations
    """

    #Open files in read mode
    predicted_file = h5py.File(prediction_path, 'r')

    #Access the data
    input_data = predicted_file['train/352/input'][indice]
    mask_1 = predicted_file['train/352/target_an'][indice]
    #predicted_mask = predicted_file['00/predicted'][indice]

    if second_mask_path:
        second_mask_file = h5py.File(second_mask_path, 'r')
        mask_2 = second_mask_file['val/352/target_an'][indice]

        legend_elements = [Line2D([0], [0], color='red', label='Predicted Mask', lw=4),
                           Line2D([0], [0], color='gold', label=r'Radiologist$_{\mathrm{O}}^{\mathrm{1}}$', lw=4),
                           Line2D([0], [0], color='turquoise', label=r'Radiologist$_{\mathrm{O}}^{\mathrm{2}}$', lw=4)]

        plt.figure(figsize=(12, 8))
        plt.imshow(input_data, cmap='gray')  # , vmin=-2, vmax=4)
        plt.colorbar()
        #plt.contourf(predicted_mask[..., 0], levels=[0.5, 1.0], alpha=0.2, colors='red')
        #plt.contour(predicted_mask[..., 0], levels=[0.5], linewidths=2.5, colors='red')
        plt.contourf(mask_1[..., 0], levels=[0.5, 1.0], alpha=0.2, colors='gold')
        plt.contour(mask_1[..., 0], levels=[0.5], linewidths=2.5, colors='gold')
        plt.contourf(mask_2[..., 0], levels=[0.5, 1.0], alpha=0.2, colors='turquoise')
        plt.contour(mask_2[..., 0], levels=[0.5], linewidths=2.5, colors='turquoise')
        #plt.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        #plt.legend(handles=legend_elements, bbox_to_anchor=(1.20, 1.0), loc='upper left')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    else:
        legend_elements = [Line2D([0], [0], color='red', label='Predicted Mask', lw=4)]#,
                           #Line2D([0], [0], color='gold', label=r'Radiologist$_{\mathrm{O}}^{\mathrm{1}}$', lw=4)]

        plt.figure(figsize=(12, 8))
        plt.imshow(input_data, cmap='gray')  # , vmin=-2, vmax=4)
        plt.colorbar()
        #plt.contourf(predicted_mask[..., 0], levels=[0.5, 1.0], alpha=0.2, colors='red')
        #plt.contour(predicted_mask[..., 0], levels=[0.5], linewidths=2.5, colors='red')
        plt.contourf(mask_1[..., 0], levels=[0.5, 1.0], alpha=0.2, colors='gold')
        plt.contour(mask_1[..., 0], levels=[0.5], linewidths=2.5, colors='gold')
        #plt.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        plt.axis('off')
        plt.tight_layout()
        plt.show()