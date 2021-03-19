import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, FixedLocator, FixedFormatter
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, KFold
import random

data_Oxy = pd.read_excel("/Volumes/LaCie/MasterThesis_Ingvild/Excel_data/200618_Inklusjonsdata_COPY_ny.xlsx", index_col=0)
data_LARC = pd.read_excel("/Volumes/LaCie/MasterThesis_Ingvild/Excel_data/150701 Kliniske data endelig versjon ny.xlsx", index_col=0)

def categorize(data):

    category = np.zeros(len(data))

    for index, row in data.iterrows():
        if row['Kjønn'] == 'M' and row['Stage'] == 2 and row['DWI'] == 'YES':
            category[index] = 1
        elif row['Kjønn'] == 'M' and row['Stage'] == 2 and row['DWI'] == 'NO':
            category[index] = 2
        elif row['Kjønn'] == 'M' and row['Stage'] == 3 and row['DWI'] == 'YES':
            category[index] = 3
        elif row['Kjønn'] == 'M' and row['Stage'] == 3 and row['DWI'] == 'NO':
            category[index] = 4
        elif row['Kjønn'] == 'M' and row['Stage'] == 4 and row['DWI'] == 'YES':
            category[index] = 5
        elif row['Kjønn'] == 'M' and row['Stage'] == 4 and row['DWI'] == 'NO':
            category[index] = 6

        elif row['Kjønn'] == 'K' and row['Stage'] == 2 and row['DWI'] == 'YES':
            category[index] = 7
        elif row['Kjønn'] == 'K' and row['Stage'] == 2 and row['DWI'] == 'NO':
            category[index] = 8
        elif row['Kjønn'] == 'K' and row['Stage'] == 3 and row['DWI'] == 'YES':
            category[index] = 9
        elif row['Kjønn'] == 'K' and row['Stage'] == 3 and row['DWI'] == 'NO':
            category[index] = 10
        elif row['Kjønn'] == 'K' and row['Stage'] == 4 and row['DWI'] == 'YES':
            category[index] = 11
        elif row['Kjønn'] == 'K' and row['Stage'] == 4 and row['DWI'] == 'NO':
            category[index] = 12

        else:
            print('Patient ' + str(index) + ' unknown category')

    return category


def plot_distribution(category, title):

    stage = FixedFormatter(['T2 \n (DWIa)', 'T2 \n (DWIna) ', 'T3 \n (DWIa)', 'T3 \n (DWIna)', 'T4 \n (DWIa)', 'T4 \n (DWIna)'])
    men = [(category == 1).sum(), (category == 2).sum(), (category == 3).sum() ,(category == 4).sum(), (category == 5).sum(), (category == 6).sum()]
    women = [(category == 7).sum(), (category == 8).sum(), (category == 9).sum(), (category == 10).sum(), (category == 11).sum(), (category == 12).sum()]

    #x = np.arange(3)
    x = np.arange(6)
    width = 0.35
    #width = 0.10
    xloc = FixedLocator(x)

    matplotlib.rcParams.update({'font.size': 25})
    matplotlib.rcParams['font.family'] = "serif"
    matplotlib.rcParams.update({'xtick.labelsize': 20})

    fig = plt.figure(figsize=(11,8))
    ax = fig.gca()

    rects1 = ax.bar(x-width/2, men, width, color='#00509E', label='Men')
    rects2 = ax.bar(x+width/2, women, width, color='#9ECFFF', label='Women')

    ax.set_ylabel(r'Number of patients')
    ax.set_xlabel(r'Stage') #, fontsize=20)
    ax.xaxis.set_major_formatter(stage)
    ax.xaxis.set_major_locator(xloc)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    for rect in rects1:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    for rect in rects2:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom, top+3)
    ax.legend()
    fig.tight_layout()
    plt.show()

#This section is needed when there is only one patient with a certain class
#E.g. in the LARC-RRP dataset the patient LARC-RRP-045 is the only one belonging to category 4

###################################################
data_LARC = data_LARC.drop(41)
data_LARC['index'] = range(0,88)
data_LARC = data_LARC.set_index('index')
##################################################

patientsOxy = pd.DataFrame(data_Oxy).to_numpy()[:,0]
categoryOxy = categorize(data_Oxy)

patientsLARC = pd.DataFrame(data_LARC).to_numpy()[:,0]
categoryLARC = categorize(data_LARC)

def traditional_split(patients, category, test_size, val_size):
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
    for trainVal_index, test_index in sss_test.split(patients, category):
        trainVal = patients[trainVal_index]
        trainVal_cat = category[trainVal_index]
        test = patients[test_index]
        test_cat = category[test_index]

    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=0)
    for train_index, val_index in sss_val.split(trainVal, trainVal_cat):
        train = trainVal[train_index]
        train_cat = trainVal_cat[train_index]
        val = trainVal[val_index]
        val_cat = trainVal_cat[val_index]

    return train, train_cat, train_index, val, val_cat, val_index, test, test_cat, test_index

def kfold_stratified_5split(patients, category, split_size):
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=split_size, random_state=0)
    for leftover_index_1, fold_index_1 in sss_test.split(patients, category):
        leftover_patients_1 = patients[leftover_index_1]
        leftover_cat_1 = category[leftover_index_1]
        patients_fold_1 = patients[fold_index_1]
        cat_fold_1 = category[fold_index_1]

    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=split_size, random_state=0)
    for leftover_index_2, fold_index_2 in sss_test.split(leftover_patients_1, leftover_cat_1):
        leftover_patients_2 = leftover_patients_1[leftover_index_2]
        leftover_cat_2 = leftover_cat_1[leftover_index_2]
        patients_fold_2 = leftover_patients_1[fold_index_2]
        cat_fold_2 = leftover_cat_1[fold_index_2]

    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=split_size, random_state=0)
    for leftover_index_3, fold_index_3 in sss_test.split(leftover_patients_2, leftover_cat_2):
        leftover_patients_3 = leftover_patients_2[leftover_index_3]
        leftover_cat_3 = leftover_cat_2[leftover_index_3]
        patients_fold_3 = leftover_patients_2[fold_index_3]
        cat_fold_3 = leftover_cat_2[fold_index_3]

    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=split_size-1, random_state=0)
    for leftover_index_4, fold_index_4 in sss_test.split(leftover_patients_3, leftover_cat_3):
        leftover_patients_4 = leftover_patients_3[leftover_index_4]
        leftover_cat_4 = leftover_cat_3[leftover_index_4]
        patients_fold_4 = leftover_patients_3[fold_index_4]
        cat_fold_4 = leftover_cat_3[fold_index_4]

    patient_folds = [patients_fold_1, patients_fold_2, patients_fold_3, patients_fold_4, leftover_patients_4]
    cat_folds = [cat_fold_1, cat_fold_2, cat_fold_3, cat_fold_4, leftover_cat_4]

    return patient_folds, cat_folds


def create_traditionalSplit_dict(train, val, test, smaller_dimensions=None):

    patient_dict = {}
    patient_dict['train'] = {}
    patient_dict['val'] = {}
    patient_dict['test'] = {}

    set_small_dimensions = set()
    set_large_dimensions = set()

    for patient in train:
        if patient in smaller_dimensions:
            set_small_dimensions.add(patient)
        else:
            set_large_dimensions.add(patient)


    if (len(set_small_dimensions)==0):
        patient_dict['train']['352'] = [set_large_dimensions]
    else:
        patient_dict['train']['352'] = [set_large_dimensions]
        patient_dict['train']['256'] = [set_small_dimensions]

    set_small_dimensions = set()
    set_large_dimensions = set()

    for patient in val:

        if patient in smaller_dimensions:
            set_small_dimensions.add(patient)
        else:
            set_large_dimensions.add(patient)

    if (len(set_small_dimensions)==0):
        patient_dict['val']['352'] = [set_large_dimensions]
    else:
        patient_dict['val']['352'] = [set_large_dimensions]
        patient_dict['val']['256'] = [set_small_dimensions]

    set_small_dimensions = set()
    set_large_dimensions = set()

    for patient in test:

        if patient in smaller_dimensions:
            set_small_dimensions.add(patient)
        else:
            set_large_dimensions.add(patient)

    if (len(set_small_dimensions)==0):
        patient_dict['test']['352'] = [set_large_dimensions]
    else:
        patient_dict['test']['352'] = [set_large_dimensions]
        patient_dict['test']['256'] = [set_small_dimensions]

    return patient_dict

def create_kfold_5split_dict(patient_folds, smaller_dimensions=None):

    patient_dict = {}
    patient_dict['fold_1'] = {}
    patient_dict['fold_2'] = {}
    patient_dict['fold_3'] = {}
    patient_dict['fold_4'] = {}
    patient_dict['fold_5'] = {}

    set_small_dimensions = set()
    set_large_dimensions = set()

    for i in range(len(patient_folds)):
        for patient in patient_folds[i]:

            if patient in smaller_dimensions:
                set_small_dimensions.add(patient)
            else:
                set_large_dimensions.add(patient)

        if (len(set_small_dimensions) == 0):
            patient_dict['fold_'+str(i+1)]['352'] = [set_large_dimensions]
        else:
            patient_dict['fold_'+str(i+1)]['352'] = [set_large_dimensions]
            patient_dict['fold_'+str(i+1)]['256'] = [set_small_dimensions]

        set_small_dimensions = set()
        set_large_dimensions = set()

    return patient_dict

trainOxy, train_catOxy, train_indexOxy, valOxy, val_catOxy, val_indexOxy, testOxy, test_catOxy, test_indexOxy = traditional_split(patientsOxy, categoryOxy, 17, 16)
trainLARC, train_catLARC, train_indexLARC, valLARC, val_catLARC, val_indexLARC, testLARC, test_catLARC, test_indexLARC = traditional_split(patientsLARC, categoryLARC, 13, 13)

trainVal_patients_Oxy = np.append(valOxy, trainOxy)
trainVal_category_Oxy = np.append(val_catOxy, train_catOxy)
patient_folds_Oxy, cat_folds_Oxy = kfold_stratified_5split(trainVal_patients_Oxy, trainVal_category_Oxy, 19)

#This section is needed when there is only one patient with a certain class

####################################################
train_catLARC = np.insert(train_catLARC,0,4)
train_indexLARC = np.insert(train_indexLARC,0,42)
trainLARC = np.insert(trainLARC, 0, 'LARC-RRP-045')
categoryLARC = np.insert(categoryLARC, 0, 4)
###################################################

#trainVal_patients_LARC = np.append(valLARC, trainLARC)
#trainVal_category_LARC = np.append(val_catLARC, train_catLARC)

def create_kfold_5splits_LARC(trainVal_patients_LARC, trainVal_category_LARC):


    temp_patients_LARC = []
    temp_cat_LARC = []
    index_array = []

    for i in range(len(trainVal_category_LARC)):
        if trainVal_category_LARC[i] in [2.,5.,9.,11.]:
            temp_patients_LARC.append(trainVal_patients_LARC[i])
            temp_cat_LARC.append(trainVal_category_LARC[i])
            index_array.append(i)

    trainVal_patients_LARC = np.delete(trainVal_patients_LARC, index_array)
    trainVal_category_LARC = np.delete(trainVal_category_LARC, index_array)

    patient_folds_LARC, cat_folds_LARC = kfold_stratified_5split(trainVal_patients_LARC, trainVal_category_LARC, 12)

    cat_folds_LARC[0] = np.append(cat_folds_LARC[0], temp_cat_LARC[0])
    patient_folds_LARC[0] = np.append(patient_folds_LARC[0], temp_patients_LARC[0])
    cat_folds_LARC[1] = np.append(cat_folds_LARC[1], temp_cat_LARC[2])
    patient_folds_LARC[1] = np.append(patient_folds_LARC[1], temp_patients_LARC[2])
    cat_folds_LARC[2] = np.append(cat_folds_LARC[2], temp_cat_LARC[4])
    patient_folds_LARC[2] = np.append(patient_folds_LARC[2], temp_patients_LARC[4])
    cat_folds_LARC[3] = np.append(cat_folds_LARC[3], temp_cat_LARC[6])
    patient_folds_LARC[3] = np.append(patient_folds_LARC[3], temp_patients_LARC[6])
    cat_folds_LARC[4] = np.append(cat_folds_LARC[4], temp_cat_LARC[10])
    patient_folds_LARC[4] = np.append(patient_folds_LARC[4], temp_patients_LARC[10])

    cat_folds_LARC[0] = np.append(cat_folds_LARC[0], temp_cat_LARC[5])
    patient_folds_LARC[0] = np.append(patient_folds_LARC[0], temp_patients_LARC[5])
    cat_folds_LARC[1] = np.append(cat_folds_LARC[1], temp_cat_LARC[11])
    patient_folds_LARC[1] = np.append(patient_folds_LARC[1], temp_patients_LARC[11])
    cat_folds_LARC[2] = np.append(cat_folds_LARC[2], temp_cat_LARC[14])
    patient_folds_LARC[2] = np.append(patient_folds_LARC[2], temp_patients_LARC[14])
    cat_folds_LARC[3] = np.append(cat_folds_LARC[3], temp_cat_LARC[3])
    patient_folds_LARC[3] = np.append(patient_folds_LARC[3], temp_patients_LARC[3])
    cat_folds_LARC[4] = np.append(cat_folds_LARC[4], temp_cat_LARC[7])
    patient_folds_LARC[4] = np.append(patient_folds_LARC[4], temp_patients_LARC[7])

    cat_folds_LARC[0] = np.append(cat_folds_LARC[0], temp_cat_LARC[9])
    patient_folds_LARC[0] = np.append(patient_folds_LARC[0], temp_patients_LARC[9])
    cat_folds_LARC[1] = np.append(cat_folds_LARC[1], temp_cat_LARC[1])
    patient_folds_LARC[1] = np.append(patient_folds_LARC[1], temp_patients_LARC[1])
    cat_folds_LARC[2] = np.append(cat_folds_LARC[2], temp_cat_LARC[8])
    patient_folds_LARC[2] = np.append(patient_folds_LARC[2], temp_patients_LARC[8])
    cat_folds_LARC[3] = np.append(cat_folds_LARC[3], temp_cat_LARC[12])
    patient_folds_LARC[3] = np.append(patient_folds_LARC[3], temp_patients_LARC[12])
    cat_folds_LARC[4] = np.append(cat_folds_LARC[4], temp_cat_LARC[13])
    patient_folds_LARC[4] = np.append(patient_folds_LARC[4], temp_patients_LARC[13])

    return patient_folds_LARC, cat_folds_LARC


#patient_folds_LARC, cat_folds_LARC = create_kfold_5splits_LARC(trainVal_patients_LARC, trainVal_category_LARC)

def combine_datasets_kfold(patient_folds1, category_folds1, patient_folds2, category_folds2):

    for i in range(len(category_folds1)):
        category_folds2[i] = np.append(category_folds1[i], category_folds2[i])
        patient_folds2[i] = np.append(patient_folds1[i], patient_folds2[i])

    return patient_folds2, category_folds2

def remove_256(dictionary):

    del dictionary['train']['256']
    del dictionary['val']['256']
    del dictionary['test']['256']

    return dictionary


#patient_folds_LARC_Oxy, cat_folds_LARC_Oxy = combine_datasets_kfold(patient_folds_Oxy, cat_folds_Oxy, patient_folds_LARC, cat_folds_LARC)
#trainVal_category_LARC_Oxy = np.append(trainVal_category_Oxy, trainVal_category_LARC)

category = np.append(categoryOxy, categoryLARC)

train = np.append(trainOxy, trainLARC)
train_cat = np.append(train_catOxy, train_catLARC)

val = np.append(valOxy, valLARC)
val_cat = np.append(val_catOxy, val_catLARC)

test = np.append(testOxy, testLARC)
test_cat = np.append(test_catOxy, test_catLARC)
"""
####### PLOT AND PRINT TRAIN, VALIDATION AND TEST ################
plot_distribution(category, r'Total dataset')
plot_distribution(train_cat, r'Training set')
plot_distribution(val_cat, r'Validation set')
plot_distribution(test_cat, r'Test set')

print(np.sort(train))
print(np.sort(val))
print(np.sort(test))


####### PLOT AND PRINT K-FOLDS ##################################
plot_distribution(trainVal_category_Oxy, r'Total dataset')
plot_distribution(cat_folds_Oxy[0], r'Total dataset')
plot_distribution(cat_folds_Oxy[1], r'Total dataset')
plot_distribution(cat_folds_Oxy[2], r'Total dataset')
plot_distribution(cat_folds_Oxy[3], r'Total dataset')
plot_distribution(cat_folds_Oxy[4], r'Total dataset')
#plot_distribution(kfold_cat_Oxy['Validation2'], r'Total dataset')
#plot_distribution(kfold_cat_Oxy['Train3'], r'Total dataset')
#plot_distribution(kfold_cat_Oxy['Validation3'], r'Total dataset')
#plot_distribution(kfold_cat_Oxy['Train4'], r'Total dataset')
#plot_distribution(kfold_cat_Oxy['Validation4'], r'Total dataset')

print('Fold1:')
print('Train:', np.sort(kfold_patients_Oxy['Fold1']['Train1']))
print('Validation:', np.sort(kfold_patients_Oxy['Fold1']['Validation1']))
print('Fold2:')
print('Train:', np.sort(kfold_patients_Oxy['Fold2']['Train2']))
print('Validation:', np.sort(kfold_patients_Oxy['Fold2']['Validation2']))
print('Fold3:')
print('Train:', np.sort(kfold_patients_Oxy['Fold3']['Train3']))
print('Validation:', np.sort(kfold_patients_Oxy['Fold3']['Validation3']))
print('Fold4:')
print('Train:', np.sort(kfold_patients_Oxy['Fold4']['Train4']))
print('Validation:', np.sort(kfold_patients_Oxy['Fold4']['Validation4']))
print('Fold5:')
print('Train:', np.sort(kfold_patients_Oxy['Fold5']['Train5']))
print('Validation:', np.sort(kfold_patients_Oxy['Fold5']['Validation5']))
"""

#kfold_patients_Oxy = convert_kFoldDictArray_to_set(kfold_patients_Oxy)


#small_dimensions_patients_Oxy = []
#tradSplit_patients_Oxy = create_traditionalSplit_dict(trainOxy, valOxy, testOxy, smaller_dimensions=small_dimensions_patients_Oxy)

small_dimensions_patients = ['LARC-RRP-011','LARC-RRP-013','LARC-RRP-014','LARC-RRP-015','LARC-RRP-016','LARC-RRP-019']
#trainLARC, valLARC, testLARC = remove_256(trainLARC, valLARC, testLARC, small_dimensions_patients)
#tradSplit_patients_LARC = create_traditionalSplit_dict(trainLARC, valLARC, testLARC, smaller_dimensions=small_dimensions_patients)
#tradSplit_patients_LARC_352 = remove_256(tradSplit_patients_LARC)

tradSplit_patients_LARC_Oxy = create_traditionalSplit_dict(train, val, test, smaller_dimensions=small_dimensions_patients)
tradSplit_patients_LARC_Oxy_352 = remove_256(tradSplit_patients_LARC_Oxy)


#kfold_patients_Oxy = create_kfold_5split_dict(patient_folds_Oxy,smaller_dimensions=small_dimensions_patients)
#kfold_patients_LARC = create_kfold_5split_dict(patient_folds_LARC,smaller_dimensions=small_dimensions_patients)
#kfold_patients_LARC_Oxy = create_kfold_5split_dict(patient_folds_LARC_Oxy,smaller_dimensions=small_dimensions_patients)



f = open("Textfiles/LARC_Oxy_tradSplit_patients_dict_352.txt", "w")
f.write(str(tradSplit_patients_LARC_Oxy_352))
f.close()

"""
f = open("LARC_Oxy_tradSplit_category.txt","w")
f.write('Total set:')
f.write(str(category))
f.write('\n')
f.write('Train split:')
f.write(str(train_cat))
f.write('\n')
f.write('Validation split:')
f.write(str(val_cat))
f.write('\n')
f.write('Test split:')
f.write(str(test_catOxy))
f.close()



f = open("Textfiles/Oxy_kfold_patients_dict.txt", "w")
f.write(str(kfold_patients_Oxy))
f.close()

f = open("Textfiles/Oxy_kfold_category.txt", "w")
f.write('Total set:')
f.write(str(cat_folds_Oxy))
f.write('\n')
f.write('fold_1:')
f.write(str(cat_folds_Oxy[0]))
f.write('\n')
f.write('fold_2:')
f.write(str(cat_folds_Oxy[1]))
f.write('\n')
f.write('fold_3:')
f.write(str(cat_folds_Oxy[2]))
f.write('\n')
f.write('fold_4:')
f.write(str(cat_folds_Oxy[3]))
f.write('\n')
f.write('fold_5:')
f.write(str(cat_folds_Oxy[4]))
f.close()
"""