
import h5py
import random
import get_data as gd
from collections import defaultdict
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from pathlib import Path
import nibabel as nib
from typing import (Callable, DefaultDict, Dict, Generator, Iterable, List, Set, Tuple, Union)

trainPatientsOxy = ['Oxytarget_24', 'Oxytarget_27', 'Oxytarget_31', 'Oxytarget_32',
 'Oxytarget_40', 'Oxytarget_44', 'Oxytarget_45', 'Oxytarget_46',
 'Oxytarget_47', 'Oxytarget_48', 'Oxytarget_50', 'Oxytarget_51',
 'Oxytarget_52', 'Oxytarget_55', 'Oxytarget_56', 'Oxytarget_57',
 'Oxytarget_58', 'Oxytarget_59', 'Oxytarget_64', 'Oxytarget_65',
 'Oxytarget_67', 'Oxytarget_68', 'Oxytarget_72', 'Oxytarget_75',
 'Oxytarget_77', 'Oxytarget_78', 'Oxytarget_79', 'Oxytarget_80',
 'Oxytarget_85', 'Oxytarget_90', 'Oxytarget_91', 'Oxytarget_95',
 'Oxytarget_96', 'Oxytarget_103', 'Oxytarget_106', 'Oxytarget_108',
 'Oxytarget_110', 'Oxytarget_113', 'Oxytarget_116', 'Oxytarget_118',
 'Oxytarget_120', 'Oxytarget_123', 'Oxytarget_125', 'Oxytarget_126',
 'Oxytarget_127', 'Oxytarget_130', 'Oxytarget_134', 'Oxytarget_143',
 'Oxytarget_144', 'Oxytarget_148', 'Oxytarget_149', 'Oxytarget_150',
 'Oxytarget_153', 'Oxytarget_154', 'Oxytarget_155', 'Oxytarget_156',
 'Oxytarget_160', 'Oxytarget_165', 'Oxytarget_166', 'Oxytarget_169',
 'Oxytarget_170', 'Oxytarget_171', 'Oxytarget_172', 'Oxytarget_173',
 'Oxytarget_174', 'Oxytarget_175', 'Oxytarget_177', 'Oxytarget_179',
 'Oxytarget_181', 'Oxytarget_185', 'Oxytarget_186', 'Oxytarget_187',
 'Oxytarget_188', 'Oxytarget_189', 'Oxytarget_190', 'Oxytarget_191',
 'Oxytarget_192']
valPatientsOxy = ['OxyTarget_028', 'OxyTarget_043', 'OxyTarget_061', 'OxyTarget_069',
 'OxyTarget_074', 'OxyTarget_094', 'OxyTarget_115', 'OxyTarget_121',
 'OxyTarget_122', 'OxyTarget_124', 'OxyTarget_133', 'OxyTarget_138',
 'OxyTarget_162', 'OxyTarget_163', 'OxyTarget_164', 'OxyTarget_184']
testPatientsOxy = ['OxyTarget_029', 'OxyTarget_041', 'OxyTarget_049', 'OxyTarget_073',
 'OxyTarget_083', 'OxyTarget_087', 'OxyTarget_088', 'OxyTarget_089',
 'OxyTarget_097', 'OxyTarget_099', 'OxyTarget_111', 'OxyTarget_128',
 'OxyTarget_131', 'OxyTarget_145', 'OxyTarget_146', 'OxyTarget_157',
 'OxyTarget_176']

trainPatientsLARC = ['LARC-RRP-001', 'LARC-RRP-003', 'LARC-RRP-006', 'LARC-RRP-007',
 'LARC-RRP-010', 'LARC-RRP-015', 'LARC-RRP-016', 'LARC-RRP-017',
 'LARC-RRP-018', 'LARC-RRP-019', 'LARC-RRP-024', 'LARC-RRP-026',
 'LARC-RRP-027', 'LARC-RRP-029', 'LARC-RRP-030', 'LARC-RRP-031',
 'LARC-RRP-033', 'LARC-RRP-035', 'LARC-RRP-036', 'LARC-RRP-037',
 'LARC-RRP-038', 'LARC-RRP-039', 'LARC-RRP-040', 'LARC-RRP-041',
 'LARC-RRP-042', 'LARC-RRP-043', 'LARC-RRP-045', 'LARC-RRP-047',
 'LARC-RRP-048', 'LARC-RRP-049', 'LARC-RRP-050', 'LARC-RRP-051',
 'LARC-RRP-052', 'LARC-RRP-053', 'LARC-RRP-055', 'LARC-RRP-058',
 'LARC-RRP-059', 'LARC-RRP-060', 'LARC-RRP-062', 'LARC-RRP-064',
 'LARC-RRP-065', 'LARC-RRP-067', 'LARC-RRP-069', 'LARC-RRP-070',
 'LARC-RRP-071', 'LARC-RRP-073', 'LARC-RRP-074', 'LARC-RRP-075',
 'LARC-RRP-076', 'LARC-RRP-077', 'LARC-RRP-078', 'LARC-RRP-079',
 'LARC-RRP-080', 'LARC-RRP-081', 'LARC-RRP-083', 'LARC-RRP-086',
 'LARC-RRP-087', 'LARC-RRP-089', 'LARC-RRP-091', 'LARC-RRP-092',
 'LARC-RRP-093', 'LARC-RRP-094', 'LARC-RRP-095']
valPatientsLARC = ['LARC-RRP-009', 'LARC-RRP-011', 'LARC-RRP-013', 'LARC-RRP-028',
 'LARC-RRP-044', 'LARC-RRP-054', 'LARC-RRP-057', 'LARC-RRP-068',
 'LARC-RRP-072', 'LARC-RRP-084', 'LARC-RRP-085', 'LARC-RRP-088',
 'LARC-RRP-090']
testPatientsLARC = ['LARC-RRP-004', 'LARC-RRP-005', 'LARC-RRP-008', 'LARC-RRP-014',
 'LARC-RRP-020', 'LARC-RRP-021', 'LARC-RRP-022', 'LARC-RRP-023',
 'LARC-RRP-032', 'LARC-RRP-034', 'LARC-RRP-066', 'LARC-RRP-096',
 'LARC-RRP-099']


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
    mask_names = ["Manual_an.nii"]

    images = np.stack([load_nii(subject / image_name).T[:,::downsample, ::downsample]
                       for image_name in image_names], axis=-1)

    masks = np.stack([load_nii(subject / mask_name).T[:,::downsample, ::downsample]
                       for mask_name in mask_names], axis=-1)

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
        chunks=(1, *shape[1:]),
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
    dataset[shape[0] :] = data
    return dataset


def get_patient_id_Oxy(patient: Path)-> int:
    """
    Return the patient ID from the path of this patient's data folder.
    """
    return int(patient.name.split('_')[1])

def get_patient_id_LARC(patient: Path)-> int:
    """
    Return the patient ID from the path of this patient's data folder.
    """
    return int(patient.name.split('-')[2])

def generate_fold_group_Oxy(
        group:h5py.Group, patient_folders:Iterable[Path])->None:
    """
    Generate dataset for a single fold.
    """
    input_dataset = None
    mask_dataset = None
    patient_ids = None
    for patient in tqdm(patient_folders):
        print(patient)
        images, masks = load_t2w_Oxy(patient)
        patient_id = get_patient_id_Oxy(patient)
        patient_id = np.ones(images.shape[0])*patient_id

        if input_dataset is None:
            input_dataset = populate_initial_dataset(images, group,"input")
            mask_dataset = populate_initial_dataset(masks, group, "target_an")
            patient_ids = populate_initial_dataset(patient_id, group, "patient_ids")
        else:
            input_dataset = extend_dataset(input_dataset, images)
            mask_dataset = extend_dataset(mask_dataset, masks)
            patient_ids = extend_dataset(patient_ids, patient_id)

def generate_fold_group_LARC(
        group:h5py.Group, patient_folders:Iterable[Path])->None:
    """
    Generate dataset for a single fold.
    """
    input_dataset = None
    mask_dataset = None
    patient_ids = None
    for patient in tqdm(patient_folders):
        print(patient)
        images, masks = load_t2w_LARC(patient)
        patient_id = get_patient_id_LARC(patient)
        patient_id = np.ones(images.shape[0])*patient_id

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
    for id_ in id_list:
        yield data_folder / f"Oxytarget_{id_}_PRE"

def patient_iter_LARC(data_folder:str, id_list:List[int])->Generator[Path,None,None]:
    """
    Iterate over the patient folders corresponding to the input id list.
    """
    data_folder = Path(data_folder)
    for id_ in id_list:

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

def generate_hdf5_file_Oxy(folds:Dict[str, List[Set[int]]], out_name:str, data_path:Path, k_fold=False, overwrite=False)->DefaultDict[str,List[str]]:
    """
    Generate a HDF5 file based on dataset splits.

    fold_names is a dictionary that maps the split names to a list of folds names in each split.
    """

    fold_names = defaultdict(list)

    out_file = data_path / out_name
    if not overwrite and out_file.is_file():
        raise RuntimeError("File exists")

    if not k_fold:
        print('Running train, val and test mode')

        with h5py.File(out_file, "w") as h5:
            for split in folds:  # split is usually train, test or val
                print(split)

                for fold in folds[split]:
                    foldname = split
                    print(foldname)

                    # Update h5py
                    group = h5.create_group(foldname)
                    fold_names[split].append(foldname)
                    fold = sorted(patient_iter_Oxy(data_path, fold))
                    generate_fold_group_Oxy(group, fold)

    else:
        print('Running k-fold mode')
        fold_num = 0
        with h5py.File(out_file, "w") as h5:
            for split in folds: #split is usually train, test or val
                print(split)

                for fold in folds[split]:
                    foldname = f"fold_{fold_num}"
                    print(foldname)
                    fold_num += 1

                    #Update h5py
                    group = h5.create_group(foldname)
                    fold_names[split].append(foldname)
                    fold = sorted(patient_iter_Oxy(data_path, fold))
                    generate_fold_group_Oxy(group, fold)

    return fold_names

def generate_hdf5_file_LARC(folds:Dict[str, List[Set[int]]], out_name:str, data_path:Path, k_fold=False, overwrite=False)->DefaultDict[str,List[str]]:
    """
    Generate a HDF5 file based on dataset splits.

    fold_names is a dictionary that maps the split names to a list of folds names in each split.
    """

    fold_names = defaultdict(list)

    out_file = data_path / out_name
    if not overwrite and out_file.is_file():
        raise RuntimeError("File exists")

    if not k_fold:
        print('Running train, val and test mode')

        with h5py.File(out_file, "w") as h5:
            for split in folds:  # split is usually train, test or val
                print(split)

                for fold in folds[split]:
                    foldname = split
                    print(foldname)

                    # Update h5py
                    group = h5.create_group(foldname)
                    fold_names[split].append(foldname)
                    fold = sorted(patient_iter_LARC(data_path, fold))
                    generate_fold_group_LARC(group, fold)

    else:
        print('Running k-fold mode')
        fold_num = 0
        with h5py.File(out_file, "w") as h5:
            for split in folds: #split is usually train, test or val
                print(split)

                for fold in folds[split]:
                    foldname = f"fold_{fold_num}"
                    print(foldname)
                    fold_num += 1

                    #Update h5py
                    group = h5.create_group(foldname)
                    fold_names[split].append(foldname)
                    fold = sorted(patient_iter_LARC(data_path, fold))
                    generate_fold_group_LARC(group, fold)

    return fold_names



splits_Oxy = create_TrainValTest_sets(trainPatientsOxy, valPatientsOxy, testPatientsOxy)
splits_LARC = create_TrainValTest_sets(trainPatientsLARC, valPatientsLARC, testPatientsLARC)

#folds = generate_folds(splits, 10)

data_path_Oxy = Path(r'/Volumes/HARDDISK/MasterThesis/Oxy_cropped')
data_path_LARC = Path(r'/Volumes/HARDDISK/MasterThesis/LARC_cropped')

generate_hdf5_file_Oxy(splits_Oxy, out_name='test_Oxy.h5', data_path=data_path_Oxy, k_fold=False, overwrite=True)
#generate_hdf5_file_LARC(splits_LARC, out_name='test_LARC.h5', data_path=data_path_LARC, k_fold=False, overwrite=True)

def print_detail(filename):
    with h5py.File(filename, 'r') as f:
        for group in f.keys():
            print(group)
            for ds_name in f[group].keys():
                print('--', ds_name, f[group][ds_name].shape)
                if ds_name == 'patient_ids':
                    print('--', np.unique(f[group][ds_name]))

print_detail('/Volumes/HARDDISK/MasterThesis/Oxy_cropped/test_Oxy.h5')

