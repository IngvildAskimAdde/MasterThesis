import os
import SimpleITK as sitk
import numpy as np
import get_data as gd
import Preprocessing as p
import useful_functions as uf



def crop_images(dataframe, new_dimension, destination_folders_list, image_filename, mask_filename, tumor_value):
    """
    The function crops the images with path given under 'imagePaths' in the dataframe to the dimension specified by new_dimension.
    The cropped images and masks are saved to the destination paths, given in destination_folders_list.

    :param dataframe: dataframe containing information of image/mask paths and dimensions
    :param new_dimension: wanted image dimensions after cropping
    :param destination_folders_list: list of destination paths to each patient
    :param image_filename: filename of image
    :param mask_filename: filename of mask
    :param tumor_value: value of tumor voxels in mask
    :return: cropped images and masks are saved to destination paths
    """
    for i in range(len(dataframe['imagePaths'])):

        image_original = sitk.ReadImage(dataframe['imagePaths'][i])
        mask_original = sitk.ReadImage(dataframe['maskPaths'][i])
        image_imsize_original = image_original.GetSize()

        if image_imsize_original[1] != 256:
            crop_start = int((image_imsize_original[1] - new_dimension) / 2)
            print(crop_start)
            crop_stop = int(image_imsize_original[1] - crop_start)
            print(crop_stop)

            image_cropped = image_original[crop_start:crop_stop, (crop_start + 10):(crop_stop + 10), :]
            mask_cropped = mask_original[crop_start:crop_stop, (crop_start + 10):(crop_stop + 10), :]

            print('Original imagesize:', dataframe['imagePaths'][i], image_original.GetSize())
            print('Original masksize:', dataframe['maskPaths'][i], mask_original.GetSize())

            print('Cropped imagesize:', image_cropped.GetSize())
            print('Cropped masksize:', mask_cropped.GetSize())

            mask_array_original = sitk.GetArrayFromImage(mask_original)
            mask_array_cropped = sitk.GetArrayFromImage(mask_cropped)

            tumor_originally = np.count_nonzero(mask_array_original.flatten() == tumor_value)
            tumor_cropped = np.count_nonzero(mask_array_cropped.flatten() == tumor_value)

            move_window = 0
            while tumor_originally != tumor_cropped:
                print('THE AMOUNT OF TUMOR IS REDUCED AFTER CROPPING! PATH:', dataframe['imagePaths'][i])

                move_window += 5

                if image_filename=='T2.nii':
                    image_cropped = image_original[(crop_start - move_window):(crop_stop - move_window),
                                    (crop_start + 10 - move_window):(crop_stop + 10 - move_window), :]
                    mask_cropped = mask_original[(crop_start - move_window):(crop_stop - move_window),
                                   (crop_start + 10 - move_window):(crop_stop + 10 - move_window), :]
                else:
                    image_cropped = image_original[(crop_start + move_window):(crop_stop + move_window),
                                    (crop_start + 10 + move_window):(crop_stop + 10 + move_window), :]
                    mask_cropped = mask_original[(crop_start + move_window):(crop_stop + move_window),
                                   (crop_start + 10 + move_window):(crop_stop + 10 + move_window), :]

                mask_array_cropped = sitk.GetArrayFromImage(mask_cropped)

                tumor_cropped = np.count_nonzero(mask_array_cropped.flatten() == tumor_value)

                print('Tumor before cropping:', tumor_originally)
                print('Tumor after moving cropping window:', tumor_cropped)

        else:
            image_cropped = image_original
            mask_cropped = mask_original

        sitk.WriteImage(image_cropped, os.path.join(destination_folders_list[i], image_filename))
        sitk.WriteImage(mask_cropped, os.path.join(destination_folders_list[i], mask_filename))

def crop_masks(dataframe, new_dimension, destination_folders_list, mask_filename, tumor_value):
    """
    The function crops the masks with path given under 'maskPaths' in the dataframe to the dimension specified by new_dimension.
    The cropped images and masks are saved to the destination paths, given in destination_folders_list.

    :param dataframe: dataframe containing information of image/mask paths and dimensions
    :param new_dimension: wanted image dimensions after cropping
    :param destination_folders_list: list of destination paths to each patient
    :param mask_filename: filename of mask
    :param tumor_value: value of tumor voxels in mask
    :return: cropped masks are saved to destination paths
    """
    for i in range(len(dataframe['maskPaths'])):

        mask_original = sitk.ReadImage(dataframe['maskPaths'][i])
        mask_imsize_original = mask_original.GetSize()

        if mask_imsize_original[1] != 256:
            crop_start = int((mask_imsize_original[1] - new_dimension) / 2)
            print(crop_start)
            crop_stop = int(mask_imsize_original[1] - crop_start)
            print(crop_stop)

            mask_cropped = mask_original[crop_start:crop_stop, (crop_start + 10):(crop_stop + 10), :]

            print('Original masksize:', dataframe['maskPaths'][i], mask_original.GetSize())

            print('Cropped masksize:', mask_cropped.GetSize())

            mask_array_original = sitk.GetArrayFromImage(mask_original)
            mask_array_cropped = sitk.GetArrayFromImage(mask_cropped)

            tumor_originally = np.count_nonzero(mask_array_original.flatten() == tumor_value)
            tumor_cropped = np.count_nonzero(mask_array_cropped.flatten() == tumor_value)

            move_window = 0
            while tumor_originally != tumor_cropped:
                print('THE AMOUNT OF TUMOR IS REDUCED AFTER CROPPING! PATH:', dataframe['maskPaths'][i])

                move_window += 5

                if mask_filename=='Manual_shh.nii' or mask_filename=='Manual_an.nii':
                    mask_cropped = mask_original[(crop_start - move_window):(crop_stop - move_window),
                                   (crop_start + 10 - move_window):(crop_stop + 10 - move_window), :]
                else:
                    mask_cropped = mask_original[(crop_start + move_window):(crop_stop + move_window),
                                   (crop_start + 10 + move_window):(crop_stop + 10 + move_window), :]

                mask_array_cropped = sitk.GetArrayFromImage(mask_cropped)

                tumor_cropped = np.count_nonzero(mask_array_cropped.flatten() == tumor_value)

                print('Tumor before cropping:', tumor_originally)
                print('Tumor after moving cropping window:', tumor_cropped)

        else:
            mask_cropped = mask_original

        sitk.WriteImage(mask_cropped, os.path.join(destination_folders_list[i], mask_filename))

def remove_nonTumor_slices(dataframe):
    """

    :param dataframe: dataframe with image paths and mask paths
    :return: saves images and masks which only contains tumor for each patient
    """

    #Iterate through each patient
    for i in range(len(dataframe['maskPaths'])):

        #Create image and mask object from file path
        image = sitk.ReadImage(dataframe['imagePaths'][i])
        mask = sitk.ReadImage(dataframe['maskPaths'][i])
        print(dataframe['imagePaths'][i])
        print(dataframe['maskPaths'][i])

        #Create image and mask as arrays
        image_array = sitk.GetArrayFromImage(image)
        mask_array = sitk.GetArrayFromImage(mask)
        slices = image.GetSize()[2] #Number of slices for the given patient

        tumor_slices = []

        #Iterate through all the image slices for a patient
        for j in range(slices):
            #If the slice contains tumor then save the slice index in the tumor_slices list
            if 1 in mask_array[j][:][:]:
                tumor_slices.append(j)

        #Get the x- and y-dimensions of the images and masks
        x_dim = int(mask_array.shape[1])
        y_dim = int(mask_array.shape[2])

        #Create new image and mask arrays with the shape of the original image and mask objects, and with
        #the same number of slices as saved in tumor_slices list
        new_mask_array = np.zeros((len(tumor_slices), x_dim, y_dim))
        new_image_array = np.zeros((len(tumor_slices), x_dim, y_dim))

        #Save the new images and masks as the original images and masks with tumor
        for k in range(len(tumor_slices)):
            new_mask_array[k][:][:] = mask_array[tumor_slices[k]][:][:]
            new_image_array[k][:][:] = image_array[tumor_slices[k]][:][:]

        #Create image and mask from new arrays
        new_mask = sitk.GetImageFromArray(new_mask_array)
        new_image = sitk.GetImageFromArray(new_image_array)

        #Save the new images and masks (only containing tumor) in the destination paths
        sitk.WriteImage(new_image, dataframe['imagePaths'][i])
        sitk.WriteImage(new_mask, dataframe['maskPaths'][i])


def crop_t2_dwi_mask(dataframe, new_dimension):#, tumor_value):#, destination_folders_list, image_filename, mask_filename, dwi_filename, tumor_value):
    """
    The function crops the images with path given under 'imagePaths' in the dataframe to the dimension specified by new_dimension.
    The cropped images and masks are saved to the destination paths, given in destination_folders_list.

    :param dataframe: dataframe containing information of image/mask paths and dimensions
    :param new_dimension: wanted image dimensions after cropping
    :param destination_folders_list: list of destination paths to each patient
    :param image_filename: filename of image
    :param mask_filename: filename of mask
    :param tumor_value: value of tumor voxels in mask
    :return: cropped images and masks are saved to destination paths
    """
    for row in range(dataframe.shape[0]):
        print(dataframe['mask'][row])
        mask_original = sitk.ReadImage(dataframe['mask'][row])
        mask_size_original = mask_original.GetSize()

        if mask_size_original[1] != 256:
            crop_start = int((mask_size_original[1] - new_dimension) / 2)
            print(crop_start)
            crop_stop = int(mask_size_original[1] - crop_start)
            print(crop_stop)

            mask_cropped = mask_original[crop_start:crop_stop, (crop_start + 10):(crop_stop + 10), :]

            print('Original masksize:', dataframe['mask'][row], mask_original.GetSize())
            print('Cropped masksize:', mask_cropped.GetSize())

            print(sitk.GetArrayFromImage(mask_original).max())
            print(np.unique(sitk.GetArrayFromImage(mask_original)))
            print(np.unique(sitk.GetArrayFromImage(mask_original)).dtype)

            mask_array_original = sitk.GetArrayFromImage(mask_original)
            mask_array_cropped = sitk.GetArrayFromImage(mask_cropped)

            #tumor_originally = np.count_nonzero(mask_array_original.flatten() == tumor_value)
            #tumor_cropped = np.count_nonzero(mask_array_cropped.flatten() == tumor_value)

        #for column in dataframe.columns[:-1]:
        #    print(dataframe[column][row])
        #t2_original = sitk.ReadImage(dataframe['t2Paths'][i])
        #dwi_original = sitk.ReadImage(dataframe['dwiPaths'][i])
        #mask_original = sitk.ReadImage(dataframe['maskPaths'][i])
        #dwi_imsize_original = dwi_original.GetSize()



if __name__ == '__main__':

    Oxy_patientPaths, Oxy_patientNames, Oxy_imagePaths, Oxy_maskPaths = gd.get_paths('/Volumes/LaCie/MasterThesis_Ingvild/Data/Oxy/TumorSlices/Oxy_cropped_TS', 'T2.nii', 'Manual_an.nii')
    Oxy_df = p.dataframe(Oxy_patientPaths, Oxy_patientNames, Oxy_imagePaths, Oxy_maskPaths)
    Oxy_df = p.dimensions(Oxy_df)

    remove_nonTumor_slices(Oxy_df)

#Oxy_patientPaths, Oxy_PatientNames, Oxy_imagePaths, Oxy_maskPaths = gd.get_paths('/Volumes/Untitled 1/Ingvild_Oxytarget', 'T2', 'an.nii')

#Create dataframe with information
#Oxy_df = p.dataframe(Oxy_patientPaths, Oxy_PatientNames, Oxy_imagePaths, Oxy_maskPaths)
#Oxy_df = p.dimensions(Oxy_df)

#Create list of destination paths
#dst_paths = uf.create_dst_paths('/Volumes/LaCie/MasterThesis_Ingvild/Oxy_cropped_corrected')

#crop_images(Oxy_df, 352, dst_paths, 'T2.nii', 'Manual_an.nii', 1000)

#uf.show_image_interactive('/Volumes/LaCie/MasterThesis_Ingvild/Oxy_cropped_corrected/Oxytarget_91_PRE/T2.nii', '/Volumes/LaCie/MasterThesis_Ingvild/Oxy_cropped_corrected/Oxytarget_91_PRE/Manual_an.nii', '2')

