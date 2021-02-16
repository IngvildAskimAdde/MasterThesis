import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, FixedLocator, FixedFormatter
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

data_Oxy = pd.read_excel("/Volumes/HARDDISK/MasterThesis/Excel_data/200618_Inklusjonsdata_COPY_ny.xlsx", index_col=0)
data_LARC = pd.read_excel("/Volumes/HARDDISK/MasterThesis/Excel_data/150701 Kliniske data endelig versjon ny.xlsx", index_col=0)

def categorize(data):

    category = np.zeros(len(data))
    """
    for index, row in data.iterrows():
        if row['Kjønn'] == 'M' and row['Stage'] == 2:
            category[index] = 1
        elif row['Kjønn'] == 'M' and row['Stage'] == 3:
            category[index] = 2
        elif row['Kjønn'] == 'M' and row['Stage'] == 4:
            category[index] = 3
        elif row['Kjønn'] == 'K' and row['Stage'] == 2:
            category[index] = 4
        elif row['Kjønn'] == 'K' and row['Stage'] == 3:
            category[index] = 5
        elif row['Kjønn'] == 'K' and row['Stage'] == 4:
            category[index] = 6
        else:
            print('Patient '+ str(index)+ ' unknown category')       
    """
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
    """
    stage = FixedFormatter(['T2', 'T3', 'T4'])
    men = [(category == 1).sum(), (category == 2).sum(), (category == 3).sum()]
    women = [(category == 4).sum(), (category == 5).sum(), (category == 6).sum()]

    stage = FixedFormatter(['T2', 'T3', 'T4'])
    men_with_dwi = [(category == 1).sum(), (category == 3).sum(), (category == 5).sum()]
    women_with_dwi = [(category == 7).sum(), (category == 9).sum(), (category == 11).sum()]
    men_without_dwi = [(category == 2).sum(), (category == 4).sum(), (category == 6).sum()]
    women_without_dwi = [(category == 8).sum(), (category == 10).sum(), (category == 12).sum()]
    """

    stage = FixedFormatter(['T2 \n (DWI available)', 'T2 ', 'T3 \n (DWI available)', 'T3 \n', 'T4 \n (DWI available)', 'T4 \n'])
    men = [(category == 1).sum(), (category == 2).sum(), (category == 3).sum() ,(category == 4).sum(), (category == 5).sum(), (category == 6).sum()]
    women = [(category == 7).sum(), (category == 8).sum(), (category == 9).sum(), (category == 10).sum(), (category == 11).sum(), (category == 12).sum()]

    #x = np.arange(3)
    x = np.arange(6)
    width = 0.35
    #width = 0.10
    xloc = FixedLocator(x)

    matplotlib.rcParams.update({'font.size': 25})
    matplotlib.rcParams['font.family'] = "serif"
    matplotlib.rcParams.update({'xtick.labelsize': 15})

    fig = plt.figure(figsize=(11,8))
    ax = fig.gca()

    rects1 = ax.bar(x-width/2, men, width, color='#2E7578', label='Men')
    rects2 = ax.bar(x+width/2, women, width, color='#97D2D4', label='Women')

    """
    rects1 = ax.bar(x-width/4, men_with_dwi, width, color='#2E7578', label='Men DWI')
    rects2 = ax.bar(x-width/2, women_with_dwi, width, color='#97D2D4', label='Women DWI')
    rects3 = ax.bar(x+width/2, men_without_dwi, width, label='Men')
    rects4 = ax.bar(x+width/4, women_without_dwi, width, label='Women')
    """
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
    """
    for rect in rects3:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    for rect in rects4:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    """
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

def kfold_split(patients, category, number_of_folds):
    skf = StratifiedKFold(n_splits=number_of_folds)

    folds_patient_dict = {}
    patient_dict = {}
    category_dict = {}

    fold = 1
    for train_index, val_index in skf.split(patients, category):
        train = patients[train_index]
        train_cat = category[train_index]
        val = patients[val_index]
        val_cat = category[val_index]

        patient_dict['Train' + str(fold)] = [set(train)]
        patient_dict['Validation' + str(fold)] = [set(val)]
        category_dict['Train' + str(fold)] = train_cat
        category_dict['Validation' + str(fold)] = val_cat

        folds_patient_dict['Fold' + str(fold)] = {}
        folds_patient_dict['Fold' + str(fold)]['Train' + str(fold)] = patient_dict['Train' + str(fold)]
        folds_patient_dict['Fold' + str(fold)]['Validation' + str(fold)] = patient_dict['Validation' + str(fold)]

        fold += 1

    return folds_patient_dict

"""
def convert_kFoldDictArray_to_set(dictionary):

    new_dict = {}

    fold = 0
    for key in dictionary:
        sub_dictionary = dictionary[key]
        for sub_key in sub_dictionary:
            sub_dictionary[sub_key] = [set(sub_dictionary[sub_key])]
        fold += 1
        new_dict['Fold' + str(fold)] = sub_dictionary

    return new_dict
"""

def create_traditionalSplit_dict(train, val, test, smaller_dimensions=None):

    patient_dict = {}
    patient_dict['Train'] = {}
    patient_dict['Validation'] = {}
    patient_dict['Test'] = {}

    set_small_dimensions = set()
    set_large_dimensions = set()

    for patient in train:

        if patient in smaller_dimensions:
            set_small_dimensions.add(patient)
        else:
            set_large_dimensions.add(patient)

    if (len(set_small_dimensions)==0):
        patient_dict['Train']['512'] = [set_large_dimensions]
    else:
        patient_dict['Train']['512'] = [set_large_dimensions]
        patient_dict['Train']['256'] = [set_small_dimensions]

    set_small_dimensions = set()
    set_large_dimensions = set()

    for patient in val:

        if patient in smaller_dimensions:
            set_small_dimensions.add(patient)
        else:
            set_large_dimensions.add(patient)

    if (len(set_small_dimensions)==0):
        patient_dict['Validation']['512'] = [set_large_dimensions]
    else:
        patient_dict['Validation']['512'] = [set_large_dimensions]
        patient_dict['Validation']['256'] = [set_small_dimensions]

    set_small_dimensions = set()
    set_large_dimensions = set()

    for patient in test:

        if patient in smaller_dimensions:
            set_small_dimensions.add(patient)
        else:
            set_large_dimensions.add(patient)

    if (len(set_small_dimensions)==0):
        patient_dict['Test']['512'] = [set_large_dimensions]
    else:
        patient_dict['Test']['512'] = [set_large_dimensions]
        patient_dict['Test']['256'] = [set_small_dimensions]

    return patient_dict

trainOxy, train_catOxy, train_indexOxy, valOxy, val_catOxy, val_indexOxy, testOxy, test_catOxy, test_indexOxy = traditional_split(patientsOxy, categoryOxy, 17, 16)
trainLARC, train_catLARC, train_indexLARC, valLARC, val_catLARC, val_indexLARC, testLARC, test_catLARC, test_indexLARC = traditional_split(patientsLARC, categoryLARC, 13, 13)

trainVal_patients_Oxy = np.append(valOxy, trainOxy)
trainVal_category_Oxy = np.append(val_catOxy, train_catOxy)
kfold_patients_Oxy = kfold_split(trainVal_patients_Oxy, trainVal_category_Oxy, 5)

#This section is needed when there is only one patient with a certain class

####################################################
train_catLARC = np.insert(train_catLARC,0,4)
train_indexLARC = np.insert(train_indexLARC,0,42)
trainLARC = np.insert(trainLARC, 0, 'LARC-RRP-045')
categoryLARC = np.insert(categoryLARC, 0, 4)
###################################################

#trainVal_patients_LARC = np.append(valLARC, trainLARC)
#trainVal_category_LARC = np.append(val_catLARC, train_catLARC)
#kfold_patients_LARC, kfold_cat_LARC = kfold_split(trainVal_patients_LARC, trainVal_category_LARC, 5)



category = np.append(categoryOxy, categoryLARC)

train = np.append(trainOxy, trainLARC)
train_cat = np.append(train_catOxy, train_catLARC)

val = np.append(valOxy, valLARC)
val_cat = np.append(val_catOxy, val_catLARC)

test = np.append(testOxy, testLARC)
test_cat = np.append(test_catOxy, test_catLARC)
"""
####### PLOT AND PRINT TRAIN, VALIDATION AND TEST ################
#plot_distribution(categoryLARC, r'Total dataset')
#plot_distribution(train_catLARC, r'Training set')
#plot_distribution(val_catLARC, r'Validation set')
#plot_distribution(test_catLARC, r'Test set')

print(np.sort(trainLARC))
print(np.sort(valLARC))
print(np.sort(testLARC))

####### PLOT AND PRINT K-FOLDS ##################################
plot_distribution(kfold_cat_Oxy['Train0'], r'Total dataset')
plot_distribution(kfold_cat_Oxy['Validation0'], r'Total dataset')
plot_distribution(kfold_cat_Oxy['Train1'], r'Total dataset')
plot_distribution(kfold_cat_Oxy['Validation1'], r'Total dataset')
plot_distribution(kfold_cat_Oxy['Train2'], r'Total dataset')
plot_distribution(kfold_cat_Oxy['Validation2'], r'Total dataset')
plot_distribution(kfold_cat_Oxy['Train3'], r'Total dataset')
plot_distribution(kfold_cat_Oxy['Validation3'], r'Total dataset')
plot_distribution(kfold_cat_Oxy['Train4'], r'Total dataset')
plot_distribution(kfold_cat_Oxy['Validation4'], r'Total dataset')

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


#kfold_patients_Oxy = convert_kFoldDictArray_to_set(kfold_patients_Oxy)


small_dimensions_patients_Oxy = []
tradSplit_patients_Oxy = create_traditionalSplit_dict(trainOxy, valOxy, testOxy, smaller_dimensions=small_dimensions_patients_Oxy)

small_dimensions_patients = ['LARC-RRP-011','LARC-RRP-013','LARC-RRP-014','LARC-RRP-015','LARC-RRP-016','LARC-RRP-019']
tradSplit_patients_LARC = create_traditionalSplit_dict(trainLARC, valLARC, testLARC, smaller_dimensions=small_dimensions_patients)

tradSplit_patients_LARC_Oxy = create_traditionalSplit_dict(train, val, test, smaller_dimensions=small_dimensions_patients)
"""

f = open("Oxy_kFold_patients_dict.txt","w")
f.write(str(kfold_patients_Oxy))
f.close()
