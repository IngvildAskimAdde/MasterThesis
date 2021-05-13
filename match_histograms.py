
from skimage.exposure import match_histograms
import SimpleITK as sitk
import useful_functions as uf
import Preprocessing as p
import os
import get_data as gd
import pandas as pd


def match_all_histograms(source_folder, destination_folder, image_filename, mask_filename, reference_image_path, DWI=False):

    dst_paths = uf.create_dst_paths(destination_folder)

    if DWI:
        patientPaths, patientNames, imagePaths, maskPaths = gd.get_paths(source_folder, image_filename, mask_filename)
        dwiPaths = uf.dwi_path(patientPaths)
        df = pd.DataFrame(dwiPaths)
        rows = df.shape[0]
        for i in range(rows):
            for column in df.columns[:-1]:
                if os.path.isfile(df[column][i]):
                    reference_path = '/Volumes/LaCie/MasterThesis_Ingvild/Data/dwi/Oxy_all_cropped_TS_updated/Oxytarget_120_PRE/' + column + '.nii'
                    reference_image = sitk.ReadImage(reference_path)
                    reference_image_array = sitk.GetArrayFromImage(reference_image)
                    print('Matching:', df[column][i], 'with', reference_path)

                    image = sitk.ReadImage(df[column][i])
                    image_array = sitk.GetArrayFromImage(image)

                    matched_array = match_histograms(image_array, reference_image_array, multichannel=False)
                    matched_image = sitk.GetImageFromArray(matched_array)

                    im_filename = column + '.nii'
                    sitk.WriteImage(matched_image, os.path.join(dst_paths[i], im_filename))
                else:
                    print(df[column][i], 'does not exist')

            print('Saving mask', df['mask'][i])
            mask = sitk.ReadImage(df['mask'][i])
            sitk.WriteImage(mask, os.path.join(dst_paths[i], mask_filename))

    else:
        df = p.create_dataframe(source_folder, image_filename, mask_filename)
        reference_image = sitk.ReadImage(reference_image_path)
        # reference_image = sitk.ReadImage('/Volumes/LaCie/MasterThesis_Ingvild/Data/Oxy/TumorSlices/Oxy_cropped_TS/Oxytarget_120_PRE/T2.nii')
        reference_image_array = sitk.GetArrayFromImage(reference_image)
        for i in range(len(df['imagePaths'])):
            print('Matching:', df['imagePaths'][i])

            image = sitk.ReadImage(df['imagePaths'][i])
            image_array = sitk.GetArrayFromImage(image)

            matched_array = match_histograms(image_array, reference_image_array, multichannel=False)

            matched_image = sitk.GetImageFromArray(matched_array)
            mask = sitk.ReadImage(df['maskPaths'][i])

            sitk.WriteImage(matched_image, os.path.join(dst_paths[i], image_filename))
            sitk.WriteImage(mask, os.path.join(dst_paths[i], mask_filename))


#match_all_histograms('/Volumes/LaCie/MasterThesis_Ingvild/Data/LARC/TumorSlices/LARC_cropped_TS',
#                     '/Volumes/LaCie/MasterThesis_Ingvild/Data/LARC/TumorSlices/LARC_cropped_TS_MHOnOxy',
#                     'image.nii', '1 RTSTRUCT LARC_MRS1-label.nii',
#                     '/Volumes/LaCie/MasterThesis_Ingvild/Data/LARC/TumorSlices/LARC_cropped_TS/LARC-RRP-003/image.nii')

match_all_histograms('/Volumes/LaCie/MasterThesis_Ingvild/Data/dwi/Oxy_all_cropped_TS_updated',
                     '/Volumes/LaCie/MasterThesis_Ingvild/Data/dwi/Oxy_all_cropped_TS_updated_MH',
                     'T2.nii', 'Manual_an.nii',
                     '/Volumes/LaCie/MasterThesis_Ingvild/Data/Oxy/TumorSlices/Oxy_cropped_TS/Oxytarget_120_PRE/T2.nii', DWI=True)

#match_all_histograms('/Volumes/LaCie/MasterThesis_Ingvild/Data/LARC_cropped',
#                     '/Volumes/LaCie/MasterThesis_Ingvild/Data/LARC_cropped_MatchedHistOnOxy',
#                     'image.nii', '1 RTSTRUCT LARC_MRS1-label.nii',
#                     '/Volumes/LaCie/MasterThesis_Ingvild/Data/Oxy_cropped_corrected/Oxytarget_120_PRE/T2.nii')




#uf.plot_matched_images('/Volumes/LaCie/MasterThesis_Ingvild/Data/LARC_cropped_TumorSlices/LARC-RRP-035/image.nii',
#                       '/Volumes/LaCie/MasterThesis_Ingvild/Data/Oxy_cropped_corrected/Oxytarget_120_PRE/T2.nii',
#                       '/Volumes/LaCie/MasterThesis_Ingvild/Data/LARC_cropped_TS_MH/LARC-RRP-035/image.nii',
#                       20, 2)