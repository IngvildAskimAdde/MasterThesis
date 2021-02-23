
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import  matplotlib

data_valfold1 = pd.read_csv('/Volumes/HARDDISK/MasterThesis/Experiments/Oxy/kfold/2d_unet_Oxy_kfold_valFold1/logs.csv')
data_valfold2 = pd.read_csv('/Volumes/HARDDISK/MasterThesis/Experiments/Oxy/kfold/2d_unet_Oxy_kfold_valFold2/logs.csv')
data_valfold3 = pd.read_csv('/Volumes/HARDDISK/MasterThesis/Experiments/Oxy/kfold/2d_unet_Oxy_kfold_valFold3/logs.csv')
data_valfold4 = pd.read_csv('/Volumes/HARDDISK/MasterThesis/Experiments/Oxy/kfold/2d_unet_Oxy_kfold_valFold4/logs.csv')
data_valfold5 = pd.read_csv('/Volumes/HARDDISK/MasterThesis/Experiments/Oxy/kfold/2d_unet_Oxy_kfold_valFold5/logs.csv')

dataframes = [data_valfold1, data_valfold2, data_valfold3, data_valfold4, data_valfold5]

def create_dataframe(dataframes_list, colname, maxsize_of_dataframe):
    dataframe = pd.DataFrame()
    dataframe['epoch'] = np.arange(0,maxsize_of_dataframe)

    for i in range(len(dataframes_list)):
        dataframe['valfold'+str(i+1)] = dataframes_list[i][colname]

    return dataframe

def plot_data(dataframe, yname):
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
    plt.title('Validation')
    plt.legend()
    fig.tight_layout()
    plt.show()

def calculate_median(dataframe):
    median = []
    std = []
    for col, value in dataframe.iteritems():
        if col != 'epoch':
            median.append(dataframe[col].median())
            std.append(dataframe[col].std())

    return median, std


dataframe = create_dataframe(dataframes, 'val_dice', 74)
plot_data(dataframe, 'Dice')
median, std = calculate_median(dataframe)



