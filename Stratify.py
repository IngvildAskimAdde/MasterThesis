import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, FixedLocator, FixedFormatter
from sklearn.model_selection import StratifiedShuffleSplit

data_Oxy = pd.read_excel("/Users/ingvildaskimadde/Documents/Skole/MaterThesis/200618_Inklusjonsdata_COPY.xlsx", index_col=0)
data_LARC = pd.read_excel("/Users/ingvildaskimadde/Documents/Skole/MaterThesis/150701 Kliniske data endelig versjon.xlsx", index_col=0)

data_1 = pd.DataFrame()
data_1['ID'] = data_Oxy['ID']
data_1['Kjønn'] = data_Oxy['Kjønn']
data_1['Stage'] = data_Oxy['Stage']

data_2 = pd.DataFrame()
data_2['ID'] = data_LARC['Database No.']
data_2['Kjønn'] = data_LARC['Kjønn']
data_2['Stage'] = data_LARC['Stage']

data = data_1.append(data_2, ignore_index=True)


def categorize(data):

    category = np.zeros(len(data))

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

    return category


def plot_distribution(category, title):

    stage = FixedFormatter(['T2', 'T3', 'T4'])
    men = [(category == 1).sum(), (category == 2).sum(), (category == 3).sum()]
    women = [(category == 4).sum(), (category == 5).sum(), (category == 6).sum()]

    x = np.arange(3)
    width = 0.35
    xloc = FixedLocator(x)

    matplotlib.rcParams.update({'font.size': 30})
    matplotlib.rcParams['font.family'] = "serif"

    fig = plt.figure(figsize=(11,8))
    ax = fig.gca()
    rects1 = ax.bar(x-width/2, men, width, color='#2E7578', label='Men')
    rects2 = ax.bar(x+width/2, women, width, color='#97D2D4', label='Women')

    ax.set_ylabel(r'Number of patients')
    ax.set_xlabel(r'Stage')
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
    ax.set_ylim(bottom, top+2)
    ax.legend()
    fig.tight_layout()
    plt.show()

#This section is needed when there is only one patient with a certain class
#E.g. in the LARC-RRP dataset the patient LARC-RRP-045 is the only one belonging to category 4

###################################################
#data = data.drop(41)
#data['index'] = range(0,88)
#data = data.set_index('index')

##################################################

patients = pd.DataFrame(data).to_numpy()[:,0]
category = categorize(data)

sss_test = StratifiedShuffleSplit(n_splits=1, test_size=30, random_state=0)
for trainVal_index, test_index in sss_test.split(patients, category):
    trainVal = patients[trainVal_index]
    trainVal_cat = category[trainVal_index]
    test = patients[test_index]
    test_cat = category[test_index]

sss_val = StratifiedShuffleSplit(n_splits=1, test_size=30, random_state=0)
for train_index, val_index in sss_val.split(trainVal, trainVal_cat):
    train = trainVal[train_index]
    train_cat = trainVal_cat[train_index]
    val = trainVal[val_index]
    val_cat = trainVal_cat[val_index]

#This section is needed when there is only one patient with a certain class

####################################################
#train_cat = np.insert(train_cat,0,4)
#train_index = np.insert(train_index,0,42)
#train = np.insert(train, 0, 'LARC-RRP-045')
#category = np.insert(category, 0, 4)
###################################################

plot_distribution(category, r'Total dataset')
#plot_distribution(train_cat, r'Training set')
#plot_distribution(val_cat, r'Validation set')
#plot_distribution(test_cat, r'Test set')

print(np.sort(train))
print(np.sort(val))
print(np.sort(test))



