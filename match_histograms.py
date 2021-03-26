
from skimage.exposure import match_histograms
import SimpleITK as sitk
import useful_functions as uf
import Preprocessing as p
import os


def match_all_histograms(source_folder, destination_folder, image_filename, mask_filename, reference_image_path):

    df = p.create_dataframe(source_folder, image_filename, mask_filename)
    dst_paths = uf.create_dst_paths(destination_folder)

    reference_image = sitk.ReadImage(reference_image_path)
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


#match_all_histograms('/Volumes/LaCie/MasterThesis_Ingvild/LARC_cropped',
#                     '/Volumes/LaCie/MasterThesis_Ingvild/LARC_cropped_MatchedHist',
#                     'image.nii', '1 RTSTRUCT LARC_MRS1-label.nii',
#                     '/Volumes/LaCie/MasterThesis_Ingvild/LARC_cropped/LARC-RRP-003/image.nii')

#uf.show_image_interactive('/Volumes/LaCie/MasterThesis_Ingvild/LARC_cropped_MatchedHist/LARC-RRP-035/image.nii',
#                          '/Volumes/LaCie/MasterThesis_Ingvild/LARC_cropped_MatchedHist/LARC-RRP-035/1 RTSTRUCT LARC_MRS1-label.nii',
#                          '1')


#uf.show_image('/Volumes/LaCie/MasterThesis_Ingvild/LARC_cropped/LARC-RRP-003/image.nii',
#              '/Volumes/LaCie/MasterThesis_Ingvild/LARC_cropped/LARC-RRP-003/1 RTSTRUCT LARC_MRS1-label.nii',
#              8)

uf.plot_matched_histograms('/Volumes/LaCie/MasterThesis_Ingvild/LARC_cropped/LARC-RRP-035/image.nii',
                       '/Volumes/LaCie/MasterThesis_Ingvild/LARC_cropped/LARC-RRP-003/image.nii',
                       '/Volumes/LaCie/MasterThesis_Ingvild/LARC_cropped_MatchedHist/LARC-RRP-035/image.nii',
                       20, 8)