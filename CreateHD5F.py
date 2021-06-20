
import h5py
import random
from collections import defaultdict, ChainMap
import numpy as np
from tqdm import tqdm
from pathlib import Path
import nibabel as nib
import ast
from typing import (Callable, DefaultDict, Dict, Generator, Iterable, List, Set, Tuple, Union)
import matplotlib.pyplot as plt


def read_dictionary(file_path):
    """
    Opens a text file and saves the content as a dictionary
    """
    file = open(file_path, 'r')
    content = file.read()
    dict = ast.literal_eval(content)
    file.close()

    return dict


def create_TrainValTest_sets(trainingPatients, validationPatients, testPatients):
    """
    Returns a dictionary with training, validation and test patient ids.
    """
    dict = {}
    training_ids = set()
    val_ids = set()
    test_ids = set()


    for patient in trainingPatients:
        if patient.startswith('Oxy'):
            patient_id = int(patient.split('_')[1])
            training_ids.add(patient_id)
        elif patient.startswith('LARC'):
            patient_id = int(patient.split('-')[2])
            training_ids.add(patient_id)
        else:
            print('Patient not recognized')

    for patient in validationPatients:
        if patient.startswith('Oxy'):
            patient_id = int(patient.split('_')[1])
            val_ids.add(patient_id)
        elif patient.startswith('LARC'):
            patient_id = int(patient.split('-')[2])
            val_ids.add(patient_id)
        else:
            print('Patient not recognized')

    for patient in testPatients:
        if patient.startswith('Oxy'):
            patient_id = int(patient.split('_')[1])
            test_ids.add(patient_id)
        elif patient.startswith('LARC'):
            patient_id = int(patient.split('-')[2])
            test_ids.add(patient_id)
        else:
            print('Patient not recognized')

    dict['train'] = [training_ids]
    dict['val'] = [val_ids]
    dict['test'] = [test_ids]

    return dict


def get_split_paths(patient, image_paths, mask_paths, list_of_patient_names):
    """
    Returns a list of paths for the patients in a given split of the dataset
    """
    image_path = None
    mask_path = None

    for j in range(len(list_of_patient_names)):
        if patient in list_of_patient_names[j]:
            image_path = (image_paths[j])
            mask_path = (mask_paths[j])

    return image_path, mask_path

def load_nii(nii_file: Union[Path, str]) -> np.array:
    """Return the contents of a nii file as a NumPy array.
    """
    return np.array(nib.load(nii_file).get_fdata())

def load_t2w_Oxy(subject:int, downsample:int=1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the images and mask for a T2W scan folder.
    """

    image_names = ["T2.nii"]
    #mask_names = ["Manual_an.nii"]
    mask_names_1 = ["Manual_an.nii"]
    mask_names_2 = ["Manual_shh.nii"]

    images = np.stack([load_nii(subject / image_name).T[:,::downsample, ::downsample]
                       for image_name in image_names], axis=-1)


    path = subject / mask_names_2[0]
    if path.exists():
        masks = np.stack([load_nii(subject / mask_name).T[:, ::downsample, ::downsample]
                           for mask_name in mask_names_2], axis=-1)
        print(path)
    else:
        masks = np.stack([load_nii(subject / mask_name).T[:, ::downsample, ::downsample]
                           for mask_name in mask_names_1], axis=-1)
        print(subject / mask_names_1[0])
    #masks = np.stack([load_nii(subject / mask_name).T[:,::downsample, ::downsample]
    #                   for mask_name in mask_names], axis=-1)

    return images, masks

def load_t2w_LARC(subject:int, downsample:int=1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the images and mask for a T2W scan folder.
    """

    image_names = ["image.nii"]
    mask_names = ["1 RTSTRUCT LARC_MRS1-label.nii"]

    images = np.stack([load_nii(subject / image_name).T[:,::downsample, ::downsample]
                       for image_name in image_names], axis=-1)

    masks = np.stack([load_nii(subject / mask_name).T[:,::downsample, ::downsample]
                       for mask_name in mask_names], axis=-1)

    print(subject / image_names[0])
    print(subject / mask_names[0])

    return images, masks

def load_mix_Oxy(subject:int, downsample:int=1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the images and mask for a T2W scan folder.
    """

    image_names = [subject / f"T2.nii"] + [subject / f"b4.nii"]
    #mask_names = ["Manual_an.nii"]
    mask_names_1 = ["Manual_an.nii"]
    mask_names_2 = ["Manual_shh.nii"]

    images = np.stack([load_nii(image_name).T[:,::downsample, ::downsample]
                       for image_name in image_names], axis=-1)


    path = subject / mask_names_2[0]
    if path.exists():
        masks = np.stack([load_nii(subject / mask_name).T[:, ::downsample, ::downsample]
                           for mask_name in mask_names_2], axis=-1)
        print(path)
    else:
        masks = np.stack([load_nii(subject / mask_name).T[:, ::downsample, ::downsample]
                           for mask_name in mask_names_1], axis=-1)
        print(subject / mask_names_1[0])
    #masks = np.stack([load_nii(subject / mask_name).T[:,::downsample, ::downsample]
    #                   for mask_name in mask_names], axis=-1)

    return images, masks




def populate_initial_dataset(data:np.ndarray, h5:h5py.Group, dataset_name:str)->h5py.Dataset:
    """
    Initialise a dataset in the input HDF5 group.
    Used to store all images in a split.
    """

    shape = data.shape
    maxshape = (None, *shape[1:])
    dataset = h5.create_dataset(
        name=dataset_name,
        data=data,
        dtype=data.dtype,
        shape=shape,
        maxshape=maxshape,
        chunks=(1,*shape[1:]),
        compression='lzf'
    )
    return dataset

def extend_dataset(dataset:h5py.Dataset, data:np.ndarray) -> h5py.Dataset:
    """
    Extend a dataset in the input HDF5 group.
    Used to update the images in a split.
    """
    shape = dataset.shape
    newshape = (dataset.shape[0] + data.shape[0], *dataset.shape[1:])
    dataset.resize(newshape)
    dataset[shape[0]:] = data
    return dataset


def get_patient_id_Oxy(patient: Path)-> int:
    """
    Return the patient ID from the path of this patient's data folder.
    """
    return (1000 + int(patient.name.split('_')[1]))

def get_patient_id_from_dict(dictionary):

    id_dict = {}

    for key, element in dictionary.items():
        id_dict[key] = {}
        for split in element:
            ids = set()
            for patient_list in dictionary[key][split]:
                for patient in patient_list:
                    if patient.startswith('Oxy'):
                        ids.add(int(patient.split(' ')[1]))
                    else:
                        ids.add(int(patient.split('-')[2]))

            id_dict[key][split] = [ids]
            ids = None

    return id_dict

def get_patient_id_LARC(patient: Path)-> int:
    """
    Return the patient ID from the path of this patient's data folder.
    """
    return int(patient.name.split('-')[2])

def generate_fold_group(
        group:h5py.Group, patient_folders:Iterable[Path])->None:
    """
    Generate dataset for a single fold.
    """
    input_dataset = None
    mask_dataset = None
    patient_ids = None
    for patient in tqdm(patient_folders):
        print(patient)
        if str(patient).endswith('PRE'):
            images, masks = load_t2w_Oxy(patient)
            patient_id = get_patient_id_Oxy(patient)
            patient_id = np.ones(images.shape[0])*patient_id
        else:
            images, masks = load_t2w_LARC(patient)
            patient_id = get_patient_id_LARC(patient)
            patient_id = np.ones(images.shape[0]) * patient_id

        if input_dataset is None:
            input_dataset = populate_initial_dataset(images, group,"input")
            mask_dataset = populate_initial_dataset(masks, group, "target_an")
            patient_ids = populate_initial_dataset(patient_id, group, "patient_ids")
        else:
            input_dataset = extend_dataset(input_dataset, images)
            mask_dataset = extend_dataset(mask_dataset, masks)
            patient_ids = extend_dataset(patient_ids, patient_id)


def generate_fold_group_mix(
        group:h5py.Group, patient_folders:Iterable[Path])->None:
    """
    Generate dataset for a single fold.
    """
    input_dataset = None
    mask_dataset = None
    patient_ids = None
    for patient in tqdm(patient_folders):
        print(patient)
        if str(patient).endswith('PRE'):
            images, masks = load_mix_Oxy(patient)
            patient_id = get_patient_id_Oxy(patient)
            patient_id = np.ones(images.shape[0])*patient_id
        else:
            images, masks = load_t2w_LARC(patient)
            patient_id = get_patient_id_LARC(patient)
            patient_id = np.ones(images.shape[0]) * patient_id

        if input_dataset is None:
            input_dataset = populate_initial_dataset(images, group,"input")
            mask_dataset = populate_initial_dataset(masks, group, "target_an")
            patient_ids = populate_initial_dataset(patient_id, group, "patient_ids")
        else:
            input_dataset = extend_dataset(input_dataset, images)
            mask_dataset = extend_dataset(mask_dataset, masks)
            patient_ids = extend_dataset(patient_ids, patient_id)


def patient_iter_Oxy(data_folder:str, id_list:List[int])->Generator[Path,None,None]:
    """
    Iterate over the patient folders corresponding to the input id list.
    """
    data_folder = Path(data_folder)
    id_list = list(id_list)
    count = 0
    for id_ in id_list:
        id_list[count] = 'O_' + str(id_)
        yield data_folder / f"Oxytarget_{id_}_PRE"

def patient_iter_LARC(data_folder:str, id_list:List[int])->Generator[Path,None,None]:
    """
    Iterate over the patient folders corresponding to the input id list.
    """
    data_folder = Path(data_folder)
    id_list = list(id_list)
    count = 0
    for id_ in id_list:
        id_list[count] = 'L_' + str(id_)
        count += 1
        if len(str(id_)) == 1:
            yield data_folder / f"LARC-RRP-00{id_}"

        elif len(str(id_)) == 2:
            yield  data_folder / f"LARC-RRP-0{id_}"

        else:
            yield data_folder / f"LARC-RRP-{id_}"

def random_split(ids:Iterable[int], n_folds:int) -> List[Set[int]]:
    """
    Randomly split the data into n equally sized folds.

    If data isn't divisible by the number of folds, then the last fold
    will have more items than the rest.
    """
    ids = list(ids)
    n_per_fold = int(len(ids) / n_folds)

    random.shuffle(ids)
    folds = [set(ids[fold_num * n_per_fold : (fold_num+1) * n_per_fold])
             for fold_num in range(n_folds)]

    missed_ids = len(ids)-n_per_fold*n_folds
    if missed_ids > 0:
        folds[-1] |= set(ids[-missed_ids:])

    return folds


def generate_folds(splits:Dict[str,Set[int]], num_per_fold:int) -> Dict[str, List[Set[int]]]:
    """
    Generate training, testing and validation folds based on input.
    """
    folds = {}
    for split, ids in splits.items():
        num_folds = int(len(ids) / num_per_fold)
        folds[split] = random_split(ids,num_folds)

    return folds

def generate_hdf5_file_Oxy(folds:Dict[str, List[Set[int]]], destination_path:str, out_name:str, data_path_Oxy:Path, data_path_LARC:Path, val=None, k_fold=False, overwrite=False)->DefaultDict[str,List[str]]:
    """
    Generate a HDF5 file based on dataset splits.

    fold_names is a dictionary that maps the split names to a list of folds names in each split.
    """

    fold_names = defaultdict(list)

    #out_file = data_path / out_name
    out_file = destination_path / out_name
    if not overwrite and out_file.is_file():
        raise RuntimeError("File exists")

    if not k_fold:
        print('Running train, val and test mode')

        with h5py.File(out_file, "w") as h5:
            for split in folds:  # split is usually train, test or val
                print(split)
                split_group = h5.create_group(split)
                for dimension in folds[split]:
                    for dimension_list in folds[split][dimension]:
                        foldname = dimension
                        print(foldname)

                        # Update h5py
                        sub_group = split_group.create_group(foldname)
                        fold_names[split].append(foldname)

                        if split=='val' and val=='LARC':
                            fold = sorted(patient_iter_LARC(data_path_LARC, dimension_list))
                        else:
                            fold = sorted(patient_iter_Oxy(data_path_Oxy, dimension_list))

                        generate_fold_group(sub_group, fold)

    else:
        print('Running k-fold mode')
        fold_num = 0
        with h5py.File(out_file, "w") as h5:
            for fold_number in folds:
                print(fold_number)
                fold_group = h5.create_group(fold_number)
                for split in folds[fold_number]: #split is usually train, test or val
                    for split_list in folds[fold_number][split]:
                        foldname = split
                        print(foldname)
                        fold_num += 1

                        #Update h5py
                        sub_group = fold_group.create_group(foldname)
                        fold_names[split].append(foldname)
                        fold = sorted(patient_iter_Oxy(data_path_Oxy, split_list))
                        generate_fold_group(sub_group, fold)

    return fold_names

def generate_hdf5_file_LARC(folds:Dict[str, List[Set[int]]], out_name:str, data_path_LARC:Path, data_path_Oxy:Path, val=None, k_fold=False, overwrite=False)->DefaultDict[str,List[str]]:
    """
    Generate a HDF5 file based on dataset splits.

    fold_names is a dictionary that maps the split names to a list of folds names in each split.
    """

    fold_names = defaultdict(list)

    out_file = data_path_LARC / out_name
    if not overwrite and out_file.is_file():
        raise RuntimeError("File exists")

    if not k_fold:
        print('Running train, val and test mode')

        with h5py.File(out_file, "w") as h5:
            for split in folds:  # split is usually train, test or val
                print(split)
                split_group = h5.create_group(split)
                for dimension in folds[split]:
                    for dimension_list in folds[split][dimension]:
                        foldname = dimension
                        print(foldname)

                        # Update h5py
                        sub_group = split_group.create_group(foldname)
                        fold_names[split].append(foldname)

                        if split=='val' and val=='Oxy':
                            fold = sorted(patient_iter_Oxy(data_path_Oxy, dimension_list))
                        else:
                            fold = sorted(patient_iter_LARC(data_path_LARC, dimension_list))

                        generate_fold_group(sub_group, fold)

    else:
        print('Running k-fold mode')
        fold_num = 0
        with h5py.File(out_file, "w") as h5:
            for fold_number in folds:
                print(fold_number)
                fold_group = h5.create_group(fold_number)
                for split in folds[fold_number]: #split is usually train, test or val
                    for split_list in folds[fold_number][split]:
                        foldname = split
                        print(foldname)
                        fold_num += 1

                        #Update h5py
                        sub_group = fold_group.create_group(foldname)
                        fold_names[split].append(foldname)
                        fold = sorted(patient_iter_LARC(data_path_LARC, split_list))
                        generate_fold_group(sub_group, fold)


    return fold_names

def generate_hdf5_file_LARC_Oxy(folds1:Dict[str, List[Set[int]]],folds2:Dict[str, List[Set[int]]], destination_path:str, out_name:str, data_path1:Path, data_path2:Path, k_fold=False, overwrite=False)->DefaultDict[str,List[str]]:
    """
    Generate a HDF5 file based on dataset splits.

    fold_names is a dictionary that maps the split names to a list of folds names in each split.
    NB! Folds2 is the dataset with the most varying image sizes!!!
    """

    fold_names = defaultdict(list)

    #out_file = data_path / out_name
    out_file = destination_path / out_name
    if not overwrite and out_file.is_file():
        raise RuntimeError("File exists")

    if not k_fold:
        print('Running train, val and test mode')

        with h5py.File(out_file, "w") as h5:
            for split in folds1:  # split is usually train, test or val
                print(split)
                split_group = h5.create_group(split)

                for dimension in folds2[split]:
                    if dimension == '352':
                        for dimension_list in folds2[split][dimension]:
                            foldname = dimension
                            print(foldname)

                            sub_group = split_group.create_group(foldname)
                            fold_names[split].append(foldname)
                            fold1 = sorted(patient_iter_LARC(data_path2, dimension_list))


                        for dimension in folds1[split]:
                            for dimension_list in folds1[split][dimension]:
                                fold2 = sorted(patient_iter_Oxy(data_path1, dimension_list))

                        fold = fold1 + fold2
                        generate_fold_group(sub_group, fold)


                    else:
                        for dimension_list in folds2[split][dimension]:
                            foldname = dimension
                            print(foldname)

                            sub_group = split_group.create_group(foldname)
                            fold_names[split].append(foldname)
                            fold = sorted(patient_iter_LARC(data_path2, dimension_list))
                            generate_fold_group(sub_group, fold)

def generate_hdf5_file_Oxy_mix(folds:Dict[str, List[Set[int]]], destination_path:str, out_name:str, data_path_Oxy:Path, data_path_LARC:Path, val=None, k_fold=False, overwrite=False)->DefaultDict[str,List[str]]:
    """
    Generate a HDF5 file based on dataset splits.

    fold_names is a dictionary that maps the split names to a list of folds names in each split.
    """

    fold_names = defaultdict(list)

    #out_file = data_path / out_name
    out_file = destination_path / out_name
    if not overwrite and out_file.is_file():
        raise RuntimeError("File exists")

    if not k_fold:
        print('Running train, val and test mode')

        with h5py.File(out_file, "w") as h5:
            for split in folds:  # split is usually train, test or val
                print(split)
                split_group = h5.create_group(split)
                for dimension in folds[split]:
                    for dimension_list in folds[split][dimension]:
                        foldname = dimension
                        print(foldname)

                        # Update h5py
                        sub_group = split_group.create_group(foldname)
                        fold_names[split].append(foldname)

                        if split=='val' and val=='LARC':
                            fold = sorted(patient_iter_LARC(data_path_LARC, dimension_list))
                        else:
                            fold = sorted(patient_iter_Oxy(data_path_Oxy, dimension_list))

                        generate_fold_group_mix(sub_group, fold)

    else:
        print('Running k-fold mode')
        fold_num = 0
        with h5py.File(out_file, "w") as h5:
            for fold_number in folds:
                print(fold_number)
                fold_group = h5.create_group(fold_number)
                for split in folds[fold_number]: #split is usually train, test or val
                    for split_list in folds[fold_number][split]:
                        foldname = split
                        print(foldname)
                        fold_num += 1

                        #Update h5py
                        sub_group = fold_group.create_group(foldname)
                        fold_names[split].append(foldname)
                        fold = sorted(patient_iter_Oxy(data_path_Oxy, split_list))
                        generate_fold_group_mix(sub_group, fold)

    return fold_names


splits_Oxy = read_dictionary('./Textfiles/Oxy_kfold_patients_DWI_Delineation2_dict.txt')
splits_ids_Oxy = get_patient_id_from_dict(splits_Oxy)

splits_LARC = read_dictionary('./Textfiles/LARC_kfold_patients_352_dict.txt')
splits_ids_LARC = get_patient_id_from_dict(splits_LARC)

data_path_Oxy = Path(r'/Volumes/LaCie/MasterThesis_Ingvild/Data/Oxy/TumorSlices/Oxy_TS_Delineation2')
data_path_LARC = Path(r'/Volumes/LaCie/MasterThesis_Ingvild/Data/LARC/TumorSlices/LARC_cropped_TS_ZScore')

#generate_hdf5_file_Oxy(splits_ids_Oxy, destination_path=Path(r'/Volumes/LaCie/MasterThesis_Ingvild/Data/Oxy/TumorSlices/Oxy_TS_Delineation2'), out_name='traditionalSplit_Oxy_TS_Delineation2.h5', data_path_Oxy=data_path_Oxy, data_path_LARC=data_path_LARC, k_fold=False, overwrite=False)
#generate_hdf5_file_LARC(splits_ids_LARC, out_name='traditionalSplit_LARC_TS.h5', data_path_LARC=data_path_LARC, data_path_Oxy=data_path_Oxy, k_fold=False, overwrite=False)
#generate_hdf5_file_LARC_Oxy(splits_ids_Oxy, splits_ids_LARC, destination_path=Path(r'/Volumes/LaCie/MasterThesis_Ingvild'), out_name='traditionalSplit_Combined_TS_MHZScore.h5', data_path1=data_path_Oxy, data_path2=data_path_LARC, k_fold=False, overwrite=False)
#generate_hdf5_file_Oxy_mix(splits_ids_Oxy, destination_path=Path(r'/Volumes/LaCie/MasterThesis_Ingvild/Data/dwi/Oxy_all_cropped'), out_name='traditionalSplit_Oxy_Mix.h5', data_path_Oxy=data_path_Oxy, data_path_LARC=data_path_LARC, k_fold=False, overwrite=False)

#generate_hdf5_file_Oxy(splits_ids_Oxy, destination_path=Path(r'/Volumes/LaCie/Franziska'), out_name='KFoldSplit_5splits_Oxy_TS_Delineation2_new.h5', data_path_Oxy=data_path_Oxy, data_path_LARC=data_path_LARC, k_fold=True, overwrite=True)
#generate_hdf5_file_LARC(splits_ids_LARC, out_name='KFoldSplit_5splits_LARC_TS_ZScore_new.h5', data_path_LARC=data_path_LARC, data_path_Oxy=data_path_Oxy, k_fold=True, overwrite=False)
#generate_hdf5_file_LARC_Oxy(splits_ids_Oxy, splits_ids_LARC,destination_path=Path(r'/Volumes/HARDDISK/MasterThesis/Oxy_cropped'), out_name='KFoldSplit_5splits_LARC_Oxy.h5',data_path1=data_path_Oxy, data_path2=data_path_LARC,k_fold=False,overwrite=True)


def print_detail(filename, k_fold=False):

    if not k_fold:
        with h5py.File(filename, 'r') as f:
            for group in f.keys():
                print(group)
                print(f[group])
                for ds_name in f[group].keys():
                    print('--', ds_name, f[group][ds_name].shape)
                #    if ds_name == 'patient_ids':
                #        print('---->', np.unique(f[group][ds_name]))

    else:
        with h5py.File(filename, 'r') as f:
            for group in f.keys():
                print(group)
                for sub_group in f[group].keys():
                    print('--', sub_group)
                    for ds_name in f[group][sub_group].keys():
                        print('----', ds_name, f[group][sub_group][ds_name].shape)
                        if ds_name == 'patient_ids':
                            print('----> Patient ids:', np.unique(f[group][sub_group][ds_name]))
                            print('----> Patient ids:', len(np.unique(f[group][sub_group][ds_name])))


def visulize_images(path_to_file, start_slice, end_slice):

    with h5py.File(path_to_file, 'r') as f:
        images = f['train/352']['input'][start_slice:end_slice]
        masks = f['train/352']['target_an'][start_slice:end_slice]
        print(f['train/352']['patient_ids'][start_slice:end_slice])

    plt.imshow(images[0][..., 0], 'gray')
    plt.contour(masks[0][..., 0], 1, levels=[0.5], colors='yellow')
    plt.show()

#print_detail('/Volumes/LaCie/MasterThesis_Ingvild/Experiments/LARC/LARC_ID_4/new_run/prediction.008.h5', k_fold=False)
#print_detail('/Volumes/LaCie/KFoldSplit_5splits_Oxy_TS_Delineation2.h5', k_fold=True)
#print_detail('/Volumes/LaCie/Franziska/KFoldSplit_5splits_Oxy_Mix_TS_ZScore_new.h5', k_fold=True)
#print_detail('/Volumes/LaCie/MasterThesis_Ingvild/Experiments/Oxy_new/Oxy_ID_28_new/external_test_valLARC_352/test/prediction_test.h5', k_fold=False)
#print_detail('/Volumes/LaCie/MasterThesis_Ingvild/Experiments/Oxy_new/Oxy_ID_39_new/prediction.072.h5', k_fold=False)

#print_detail('/Volumes/HARDDISK/MasterThesis/Oxy_cropped/KFoldSplit_5splits_Oxy.h5', k_fold=True)
#print_detail('/Volumes/HARDDISK/MasterThesis/LARC_cropped/KFoldSplit_5splits_LARC.h5', k_fold=True)
#print_detail('/Volumes/HARDDISK/MasterThesis/Oxy_cropped/KFoldSplit_5splits_LARC_Oxy.h5', k_fold=True)

#print_detail('/Volumes/LaCie/MasterThesis_Ingvild/Experiments/Combined_new/Combined_ID_23_new/prediction.031.h5', k_fold=False)

#visulize_images('/Volumes/LaCie/MasterThesis_Ingvild/HDF5_data/traditionalSplit_Oxy_new.h5',66 ,100)

