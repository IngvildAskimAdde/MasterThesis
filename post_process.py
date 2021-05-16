"""
@author: huynhngoc
"""

import customize_obj


if __name__ == '__main__':
    output_folder = '/Volumes/LaCie/MasterThesis_Ingvild/Experiments/LARC/LARC_ID_55/' # change this to the folder you want to store the result
    dataset_file = '/Volumes/LaCie/MasterThesis_Ingvild/HDF5_data/traditionalSplit_LARC_TS.h5' # path to the dataset

    predicted_h5 = '/Volumes/LaCie/MasterThesis_Ingvild/Experiments/LARC/LARC_ID_55/prediction.038.h5' # the prediction file you want to calculate the dice

    dice_per_slice = output_folder + 'slice_352.csv'
    dice_per_patient = output_folder + 'patient_352.csv'
    merge_file = output_folder + 'merge_images_352.h5'

    customize_obj.H5MetaDataMapping(
        dataset_file,
        dice_per_slice,
        folds=['val/352'], # change this to ['test'] if you want to calculate the dice of the test prediction
        fold_prefix='',
        dataset_names=['patient_ids'] #, 'slice_idx']
    ).post_process()

    customize_obj.H5CalculateFScore(
        predicted_h5,
        dice_per_slice
    ).post_process()

    customize_obj.H5Merge2dSlice(
        predicted_h5,
        dice_per_slice,
        map_column='patient_ids',
        merge_file=merge_file,
        save_file=dice_per_patient
    ).post_process()

    customize_obj.H5CalculateFScore(
        merge_file,
        dice_per_patient,
        map_file=dice_per_patient,
        map_column='patient_ids'
    ).post_process()