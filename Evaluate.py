
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import  matplotlib

ID = '54'
folder_name = 'LARC'
dataset = 'LARC'

LARC = pd.read_csv('/Volumes/LaCie/MasterThesis_Ingvild/Experiments/LARC/LARC_ID_' + ID + '/logs.csv')
#Oxy = pd.read_csv('/Volumes/LaCie/MasterThesis_Ingvild/Experiments/Oxy_new/Oxy_ID_' + ID + '/logs.csv')
#Oxy_ID_8_new = pd.read_csv('/Users/ingvildaskimadde/Desktop/Oxy_ID_8_new/logs.csv')
#Combined = pd.read_csv('/Volumes/LaCie/MasterThesis_Ingvild/Experiments/Combined_new/Combined_ID_' + ID + '/logs.csv')

#Oxy_patient = pd.read_csv('/Volumes/LaCie/MasterThesis_Ingvild/Experiments/Oxy_new/Oxy_ID_' + ID + '/patient.csv')
#Oxy_slice = pd.read_csv('/Volumes/LaCie/MasterThesis_Ingvild/Experiments/Oxy_new/Oxy_ID_' + ID + '/slice.csv')
#slice_352 = pd.read_csv('/Volumes/LaCie/MasterThesis_Ingvild/Experiments/' + folder_name + '/' + dataset + '_ID_' + ID + '/slice_352.csv')
#slice_256 = pd.read_csv('/Volumes/LaCie/MasterThesis_Ingvild/Experiments/' + folder_name + '/' + dataset + '_ID_' + ID + '/slice_256.csv')
#slice = slice_352.append(slice_256)

#patient = pd.read_csv('/Users/ingvild/Documents/MasterThesis/Experiments/LARC_ID_52/patient.csv')
#slice = pd.read_csv('/Users/ingvild/Documents/MasterThesis/Experiments/LARC_ID_52/slice.csv')

patient_352 = pd.read_csv('/Volumes/LaCie/MasterThesis_Ingvild/Experiments/' + folder_name + '/' + dataset + '_ID_' + ID + '/patient_352.csv')
patient_256 = pd.read_csv('/Volumes/LaCie/MasterThesis_Ingvild/Experiments/' + folder_name + '/' + dataset + '_ID_' + ID + '/patient_256.csv')
patient = patient_352.append(patient_256)

#dataframes = [data_valfold1, data_valfold2, data_valfold3, data_valfold4, data_valfold5]

def create_dataframe(dataframes_list, colname, maxsize_of_dataframe):
    """
    Takes in a list of dataframes and extracts the information from a column given by the parameter colname.
    The information in the columns with colname is merged into one dataframe, which is returned.
    """
    dataframe = pd.DataFrame()
    dataframe['epoch'] = np.arange(0,maxsize_of_dataframe)

    for i in range(len(dataframes_list)):
        dataframe['valfold'+str(i+1)] = dataframes_list[i][colname]

    return dataframe

def plot_data(dataframe, yname):
    """
    Plots the data given in the dataframe.
    yname: label of y-axis
    """
    number_of_epochs = np.size(dataframe, axis=0)
    number_of_plots = np.size(dataframe, axis=1)-1
    x_axis = np.arange(0,number_of_epochs)

    matplotlib.rcParams.update({'font.size': 20})
    matplotlib.rcParams['font.family'] = "serif"
    fig = plt.figure(figsize=(11,8))
    count = 0
    for col, item in dataframe.iteritems():
        if col != 'epoch':
            plt.plot(x_axis, dataframe[col], label='Validation = Fold '+str(count))
        count += 1

    plt.xlabel('Epoch')
    plt.ylabel(yname)
    plt.ylim(0,1)
    #plt.title('Validation')
    plt.legend()
    fig.tight_layout()
    plt.show()

def calculate_median(dataframe):
    """
    Returns a list of median and standard deviation values of the columns in a dataframe.
    """
    median = []
    std = []
    for col, value in dataframe.iteritems():
        if col != 'epoch':
            median.append(dataframe[col].median())
            std.append(dataframe[col].std())

    return median, std

def find_best_epoch(dataframe):

    max_dice = dataframe['val_dice'].max()
    max_dice_index = dataframe[dataframe['val_dice'] == dataframe['val_dice'].max()].index.values
    best_epoch = max_dice_index+1
    #min_loss = dataframe['val_loss'].min()
    #min_loss_index = dataframe[dataframe['val_loss'] == dataframe['val_loss'].min()].index.values
    print('Best Dice:', max_dice)
    return max_dice, best_epoch

def get_data(dataframe, column):
    print(dataframe.describe())
    print('Mean:', dataframe[column].mean())
    print('Std:', dataframe[column].std())
    return dataframe[column].mean(), dataframe[column].median()



#dataframe = create_dataframe(dataframes, 'val_dice', 93)
#plot_data(dataframe, 'Dice')
#median, std = calculate_median(dataframe)

#max_dice, epoch = find_best_epoch(Combined_ID_9)
max_dice, epoch = find_best_epoch(LARC)
mean, median = get_data(patient, 'f1_score')
#mean = get_data(LARC_ID_10_slice, 'f1_score')



