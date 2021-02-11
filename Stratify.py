import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, FixedLocator, FixedFormatter
from sklearn.model_selection import StratifiedShuffleSplit

data_Oxy = pd.read_excel("/Users/ingvildaskimadde/Documents/Skole/MaterThesis/200618_Inklusjonsdata_COPY.xlsx", index_col=0)
data_LARC = pd.read_excel("/Users/ingvildaskimadde/Documents/Skole/MaterThesis/150701 Kliniske data endelig versjon.xlsx", index_col=0)

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

def split(patients, category, test_size, val_size):
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

trainOxy, train_catOxy, train_indexOxy, valOxy, val_catOxy, val_indexOxy, testOxy, test_catOxy, test_indexOxy = split(patientsOxy, categoryOxy, 17, 16)
trainLARC, train_catLARC, train_indexLARC, valLARC, val_catLARC, val_indexLARC, testLARC, test_catLARC, test_indexLARC = split(patientsLARC, categoryLARC, 13, 13)

#This section is needed when there is only one patient with a certain class

####################################################
train_catLARC = np.insert(train_catLARC,0,4)
train_indexLARC = np.insert(train_indexLARC,0,42)
trainLARC = np.insert(trainLARC, 0, 'LARC-RRP-045')
categoryLARC = np.insert(categoryLARC, 0, 4)
###################################################

category = np.append(categoryOxy, categoryLARC)

train = np.append(trainOxy, trainLARC)
train_cat = np.append(train_catOxy, train_catLARC)

val = np.append(valOxy, valLARC)
val_cat = np.append(val_catOxy, val_catLARC)

test = np.append(testOxy, testLARC)
test_cat = np.append(test_catOxy, test_catLARC)


plot_distribution(category, r'Total dataset')
plot_distribution(train_cat, r'Training set')
plot_distribution(val_cat, r'Validation set')
plot_distribution(test_cat, r'Test set')

print(np.sort(train))
print(np.sort(val))
print(np.sort(test))
