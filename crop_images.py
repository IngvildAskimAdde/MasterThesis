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
        mask_an = sitk.ReadImage(dataframe['maskPaths'][i])

        path_shh = dataframe['patientPaths'][i] + '/Manual_shh.nii'
        print(path_shh)
        mask_shh = sitk.ReadImage(path_shh)
        print(dataframe['imagePaths'][i])
        print(dataframe['maskPaths'][i])

        #Create image and mask as arrays
        image_array = sitk.GetArrayFromImage(image)
        mask_array_an = sitk.GetArrayFromImage(mask_an)
        mask_array_shh = sitk.GetArrayFromImage(mask_shh)
        slices = image.GetSize()[2] #Number of slices for the given patient

        tumor_slices = []

        #Iterate through all the image slices for a patient
        for j in range(slices):
            #If the slice contains tumor then save the slice index in the tumor_slices list
            if 1 in mask_array_an[j][:][:]:
                if dataframe['ID'][i] == 'LARC-RRP-033' and j == 10: #Do not include this slice (noise from radiologist)
                    break
                else:
                    tumor_slices.append(j)

        #Get the x- and y-dimensions of the images and masks
        x_dim = int(mask_array_an.shape[1])
        y_dim = int(mask_array_an.shape[2])

        #Create new image and mask arrays with the shape of the original image and mask objects, and with
        #the same number of slices as saved in tumor_slices list
        new_mask_array_an = np.zeros((len(tumor_slices), x_dim, y_dim))
        new_mask_array_shh = np.zeros((len(tumor_slices), x_dim, y_dim))
        new_image_array = np.zeros((len(tumor_slices), x_dim, y_dim))

        #Save the new images and masks as the original images and masks with tumor
        for k in range(len(tumor_slices)):
            new_mask_array_an[k][:][:] = mask_array_an[tumor_slices[k]][:][:]
            new_mask_array_shh[k][:][:] = mask_array_shh[tumor_slices[k]][:][:]
            new_image_array[k][:][:] = image_array[tumor_slices[k]][:][:]

        #Create image and mask from new arrays
        new_mask_an = sitk.GetImageFromArray(new_mask_array_an)
        new_mask_shh = sitk.GetImageFromArray(new_mask_array_shh)
        new_image = sitk.GetImageFromArray(new_image_array)

        #Save the new images and masks (only containing tumor) in the destination paths
        sitk.WriteImage(new_image, dataframe['imagePaths'][i])
        sitk.WriteImage(new_mask_an, dataframe['maskPaths'][i])
        sitk.WriteImage(new_mask_shh, path_shh)


def crop_t2_dwi_mask(dataframe, new_dimension, tumor_value, destination_folders_list, image_filename, mask_filename):
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

            images = {'b0': [], 'b1': [], 'b2': [], 'b3': [], 'b4': [], 'b5': [], 'b6': [], 't2':[]}
            for column in dataframe.columns[:-1]:
                if os.path.isfile(dataframe[column][row]):
                    print(dataframe[column][row])
                    image_original = sitk.ReadImage(dataframe[column][row])
                    print('Original DWI size:', image_original.GetSize())
                    image_cropped = image_original[crop_start:crop_stop, (crop_start + 10):(crop_stop + 10), :]
                    images[column] = image_cropped
                    print('New DWI size:', images[column].GetSize())
                else:
                    print(dataframe[column][row], 'does not exist')
                    images[column] = None

            print('Original masksize:', dataframe['mask'][row], mask_original.GetSize())
            print('Cropped masksize:', mask_cropped.GetSize())

            mask_array_original = sitk.GetArrayFromImage(mask_original)
            mask_array_cropped = sitk.GetArrayFromImage(mask_cropped)

            tumor_originally = np.count_nonzero(mask_array_original.flatten() == tumor_value)
            tumor_cropped = np.count_nonzero(mask_array_cropped.flatten() == tumor_value)

            move_window = 0
            while tumor_originally != tumor_cropped:
                print('THE AMOUNT OF TUMOR IS REDUCED AFTER CROPPING! PATH:', dataframe['mask'][row])

                move_window += 5

                if image_filename == 'T2.nii':
                    mask_cropped = mask_original[(crop_start - move_window):(crop_stop - move_window),
                                   (crop_start + 10 - move_window):(crop_stop + 10 - move_window), :]
                    for column in dataframe.columns[:-1]:
                        if os.path.isfile(dataframe[column][row]):
                            print(dataframe[column][row])
                            image_original = sitk.ReadImage(dataframe[column][row])
                            print('Original DWI size:', image_original.GetSize())
                            image_cropped = image_original[(crop_start - move_window):(crop_stop - move_window),
                                   (crop_start + 10 - move_window):(crop_stop + 10 - move_window), :]
                            images[column] = image_cropped
                            print('New DWI size:', images[column].GetSize())
                        else:
                            print(dataframe[column][row], 'does not exist')
                            images[column] = None
                else:
                    mask_cropped = mask_original[(crop_start + move_window):(crop_stop + move_window),
                                   (crop_start + 10 + move_window):(crop_stop + 10 + move_window), :]
                    for column in dataframe.columns[:-1]:
                        if os.path.isfile(dataframe[column][row]):
                            print(dataframe[column][row])
                            image_original = sitk.ReadImage(dataframe[column][row])
                            print('Original DWI size:', image_original.GetSize())
                            image_cropped = image_original[(crop_start + move_window):(crop_stop + move_window),
                                   (crop_start + 10 + move_window):(crop_stop + 10 + move_window), :]
                            images[column] = image_cropped
                            print('New DWI size:', images[column].GetSize())
                        else:
                            print(dataframe[column][row], 'does not exist')
                            images[column] = None

                mask_array_cropped = sitk.GetArrayFromImage(mask_cropped)

                tumor_cropped = np.count_nonzero(mask_array_cropped.flatten() == tumor_value)

                print('Tumor before cropping:', tumor_originally)
                print('Tumor after moving cropping window:', tumor_cropped)

        sitk.WriteImage(mask_cropped, os.path.join(destination_folders_list[row], mask_filename))

        for key in images:
            if images[key] == None:
                print('None')
            else:
                im_filename = key + '.nii'
                sitk.WriteImage(images[key], os.path.join(destination_folders_list[row], im_filename))


def remove_nonTumor_slices_dwi(dataframe, dst_folders):
    """

    :param dataframe: dataframe with image paths and mask paths
    :return: saves images and masks which only contains tumor for each patient
    """

    #Iterate through each patient
    for i in range(len(dataframe['mask'])):

        #Create mask object from file path
        mask = sitk.ReadImage(dataframe['mask'][i])
        print(dataframe['mask'][i])

        #Create image and mask as arrays
        mask_array = sitk.GetArrayFromImage(mask)
        slices = mask.GetSize()[2] #Number of slices for the given patient

        tumor_slices = []

        #Iterate through all the image slices for a patient
        for j in range(slices):
            #If the slice contains tumor then save the slice index in the tumor_slices list
            if 1 in mask_array[j][:][:]:
                #if dataframe['ID'][i] == 'LARC-RRP-033' and j == 10: #Do not include this slice (noise from radiologist)
                #    break
                #else:
                tumor_slices.append(j)

        #Get the x- and y-dimensions of the images and masks
        x_dim = int(mask_array.shape[1])
        y_dim = int(mask_array.shape[2])

        #Create new image and mask arrays with the shape of the original image and mask objects, and with
        #the same number of slices as saved in tumor_slices list
        new_mask_array = np.zeros((len(tumor_slices), x_dim, y_dim))
        new_image_array = np.zeros((len(tumor_slices), x_dim, y_dim))

        for k in range(len(tumor_slices)):
            new_mask_array[k][:][:] = mask_array[tumor_slices[k]][:][:]

        # Create mask from new arrays
        new_mask = sitk.GetImageFromArray(new_mask_array)

        # Save the masks (only containing tumor) in the destination paths
        sitk.WriteImage(new_mask, dataframe['mask'][i])

        for column in dataframe.columns[:-1]:
            print(dataframe[column][i])
            if os.path.isfile(dataframe[column][i]):
                image = sitk.ReadImage(dataframe[column][i])
                print('Original image size:', image.GetSize())
                image_array = sitk.GetArrayFromImage(image)
                for k in range(len(tumor_slices)):
                    new_mask_array[k][:][:] = mask_array[tumor_slices[k]][:][:]
                    new_image_array[k][:][:] = image_array[tumor_slices[k]][:][:]

                new_image = sitk.GetImageFromArray(new_image_array)
                print('New image size:', new_image.GetSize())
                im_filename = column + '.nii'
                sitk.WriteImage(new_image, os.path.join(dst_folders[i], im_filename))
            else:
                print(dataframe[column][i], 'does not exist')


def remove_rotated_dwi(dataframe, dst_folders):
    """
    The function removes the slices where the DW images does not cover the tumor (due to slightly different orientations
    of DWI and T2).
    :param dataframe: dataframe of paths to all DW images, T2 image and manual delineation (mask) for each patient
    :param dst_folders: list of paths to folders of patients where the new images will be saved
    :return: DW images, T2 image and manual delineation where the slices that the DW images do not cover the entire
    tumor is removed
    """

    #Iterate through each patient
    for row in range(dataframe.shape[0]):

        #Create mask array
        mask = sitk.ReadImage(dataframe['mask'][row])
        mask_array = sitk.GetArrayFromImage(mask)

        #Iterate over all image files for the patient
        for column in dataframe.columns[:-2]:
            print(dataframe[column][row])

            #Check if the image file exists
            if os.path.isfile(dataframe[column][row]):

                #Create image array
                image = sitk.ReadImage(dataframe[column][row])
                image_array = sitk.GetArrayFromImage(image)

                #Go through each image slice and check if the tumor is outside of the DW image.
                #If the tumor is outside of the DW image then the slice is not added to the slices_to_keep list.
                #If the tumor is inside of the DW image then the slice is added to the slices_to_keep list.
                slices_to_keep = []
                keep = 0
                for slice in range(image_array.shape[0]):
                    for i in range(image_array.shape[1]):
                        for j in range(image_array.shape[2]):
                            pixel_im = image_array[slice][i][j]
                            pixel_mask = mask_array[slice][i][j]
                            if pixel_im == 0 and pixel_mask == 1:
                                print('Tumor outside of DWI', slice)
                                keep = 1
                                break
                        else:
                            continue
                        break

                    if keep == 1:
                        print('Not adding slice number', slice)
                    else:
                        slices_to_keep.append(slice)
                    keep = 0
            else:
                print(dataframe[column][row], 'does not exist')

        #Print which slices we want to keep
        print(slices_to_keep)

        #Get the x- and y-dimensions of the images and masks
        x_dim = int(mask_array.shape[1])
        y_dim = int(mask_array.shape[2])

        #Create new image and mask arrays with the shape of the original image and mask objects, and with
        #the same number of slices as saved in tumor_slices list
        new_mask_array = np.zeros((len(slices_to_keep), x_dim, y_dim))
        new_image_array = np.zeros((len(slices_to_keep), x_dim, y_dim))

        for k in range(len(slices_to_keep)):
            new_mask_array[k][:][:] = mask_array[slices_to_keep[k]][:][:]

        #Create mask from new arrays
        new_mask = sitk.GetImageFromArray(new_mask_array)

        #Save the masks (only containing tumor) in the destination paths
        sitk.WriteImage(new_mask, dataframe['mask'][row])

        #Create new images for all of the image files, and save them
        for column in dataframe.columns[:-1]:
            print(dataframe[column][row])
            if os.path.isfile(dataframe[column][row]):
                image = sitk.ReadImage(dataframe[column][row])
                print('Original image size:', image.GetSize())
                image_array = sitk.GetArrayFromImage(image)
                for k in range(len(slices_to_keep)):
                    new_image_array[k][:][:] = image_array[slices_to_keep[k]][:][:]

                new_image = sitk.GetImageFromArray(new_image_array)
                print('New image size:', new_image.GetSize())
                im_filename = column + '.nii'
                sitk.WriteImage(new_image, os.path.join(dst_folders[row], im_filename))
            else:
                print(dataframe[column][row], 'does not exist')


if __name__ == '__main__':

    #Oxy_patientPaths, Oxy_patientNames, Oxy_imagePaths, Oxy_maskPaths = gd.get_paths('/Volumes/LaCie/MasterThesis_Ingvild/Data/LARC/TumorSlices/LARC_cropped_TS_new', 'image.nii', '1 RTSTRUCT LARC_MRS1-label.nii')
    Oxy_patientPaths, Oxy_PatientNames, Oxy_imagePaths, Oxy_maskPaths = gd.get_paths('/Volumes/LaCie/MasterThesis_Ingvild/Data/dwi/Oxy_all_cropped', 'b4', 'an.nii')
    Oxy_df = p.dataframe(Oxy_patientPaths, Oxy_PatientNames, Oxy_imagePaths, Oxy_maskPaths)
    Oxy_df = p.dimensions(Oxy_df)

    #remove_nonTumor_slices(Oxy_df)

#Oxy_patientPaths, Oxy_PatientNames, Oxy_imagePaths, Oxy_maskPaths = gd.get_paths('/Volumes/Untitled 1/Ingvild_Oxytarget', 'T2', 'an.nii')

#Create dataframe with information
#Oxy_df = p.dataframe(Oxy_patientPaths, Oxy_PatientNames, Oxy_imagePaths, Oxy_maskPaths)
#Oxy_df = p.dimensions(Oxy_df)

#Create list of destination paths
#dst_paths = uf.create_dst_paths('/Volumes/LaCie/MasterThesis_Ingvild/Oxy_cropped_corrected')

#crop_images(Oxy_df, 352, dst_paths, 'T2.nii', 'Manual_an.nii', 1000)

#uf.show_image_interactive('/Volumes/LaCie/MasterThesis_Ingvild/Oxy_cropped_corrected/Oxytarget_91_PRE/T2.nii', '/Volumes/LaCie/MasterThesis_Ingvild/Oxy_cropped_corrected/Oxytarget_91_PRE/Manual_an.nii', '2')

