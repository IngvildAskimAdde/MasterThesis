
import useful_functions as uf
import ImageViewer as iv
import SimpleITK as sitk
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
"""
image_path = '/Volumes/Untitled 1/Ingvild_Oxytarget/Oxytarget_74_PRE/T2.nii'
mask_path = '/Volumes/Untitled 1/Ingvild_Oxytarget/Oxytarget_74_PRE/Manual_an.nii'

image = sitk.ReadImage(image_path)
mask = sitk.ReadImage(mask_path)

v = iv.Viewer(view_mode='1', mask_to_show=['a'])
v.set_image(image, label='image')
v.set_mask(mask, label='mask', color_rgb=[60, 180, 75])
v.show()


uf.show_image_interactive('/Volumes/Untitled 1/Ingvild_Oxytarget/Oxytarget_74_PRE/T2.nii',
                         '/Volumes/Untitled 1/Ingvild_Oxytarget/Oxytarget_74_PRE/Manual_an.nii',
                          '1')

uf.show_image('/Volumes/Untitled 1/Ingvild_Oxytarget/Oxytarget_74_PRE/T2.nii',
              '/Volumes/Untitled 1/Ingvild_Oxytarget/Oxytarget_74_PRE/Manual_an.nii',
              6)

path = '/Volumes/LaCie/MasterThesis_Ingvild/Experiments/Oxy_new/Oxy_ID_6_new/prediction.085.h5'
#path = '/Volumes/LaCie/MasterThesis_Ingvild/HDF5_data/traditionalSplit_Oxy_corrected.h5'
indice = 354
file = h5py.File(path,'r')
#data = file['val/352/input'][indice]
data = file['x'][indice]
#mask = file['val/352/target_an'][indice]
mask = file['y'][indice]
predicted = file['predicted'][indice]
#patient = file['val/352/patient_ids'][indice]
#print(data)
#print(data.shape)
#print(data[0][...,0].shape)
#print(data.max())
#print(mask.max())
#print(patient)

plt.figure(figsize=(11,8))
plt.imshow(data, cmap='gray')#, vmin=-2, vmax=4)
plt.contourf(predicted[...,0], levels=[0.5,1.0], alpha=0.5, colors='#F58230')
plt.contour(predicted[...,0], levels=[0.5], linewidths=2.5, colors='#F58230')
plt.contourf(mask[...,0], levels=[0.5,1.0], alpha=0.25, colors='#3cb44b')
plt.contour(mask[...,0], levels=[0.5], linewidths=2.5, colors='#3cb44b')
plt.axis('off')
plt.show()


def calculate_dice(mask_a, mask_b):
    
    Calculate DICE score for two binary masks (=sitk images)
    

    dice = 2*np.count_nonzero(mask_a & mask_b) / (np.count_nonzero(mask_a) + np.count_nonzero(mask_b))
    return dice


path = '/Volumes/LaCie/MasterThesis_Ingvild/Experiments/Oxy_new/Oxy_ID_6_new/prediction.085.h5'
file = h5py.File(path,'r')
masks = file['y']
dice = []
for i in range(masks.shape[0]):

    true_mask_i = sitk.GetImageFromArray(masks[i])
    true_mask_img = sitk.ReadImage(true_mask_i) > 0
    true_mask_arr = sitk.GetArrayFromImage(true_mask_img)
    print(true_mask_arr)
    #predicted_mask = file['predicted'][i] > 0
    #dice_i = calculate_dice(true_mask, predicted_mask)
    #dice.append(dice_i)
#print(dice)
"""


def main_aug():

    excel_path = '/Volumes/LaCie/MasterThesis_Ingvild/Excel_data/Experiment_plan.xlsx'

    # Define correct experiments (IDs)
    ids_LARC = [24, 27, 28]
    ids_Oxy = [16, 17, 18]
    ids_Comb = [15, 16, 17]

    # Creating dataframes of det f1 scores of the validation patients
    Oxy = uf.dataframe_of_f1scores(excel_path, 'Oxy_new', ['patient.csv'], ids_Oxy)
    LARC = uf.dataframe_of_f1scores(excel_path, 'LARC', ['patient_352.csv', 'patient_256.csv'], ids_LARC)
    Combined = uf.dataframe_of_f1scores(excel_path, 'Combined_new', ['patient_352.csv', 'patient_256.csv'], ids_Comb)

    # Creating dictionary of dataframes
    dictionary = {'OxyTarget': Oxy, 'LARC-RRP': LARC, 'Combined': Combined}

    col_names_aug = ['No', 'Default', 'Best Combination']
    col_names_norm = ['No', 'Z-Score', 'Matched Hist', 'Matched Hist + Z-Score']

    for key in dictionary:
        print(key)
        for i in range(len(dictionary[key].columns)):
            dictionary[key].columns.values[i] = col_names_aug[i]
            print(dictionary[key].median())
        dictionary[key]['Data'] = key
        dictionary[key] = pd.melt(dictionary[key], id_vars=['Data'], var_name=['Parameters'])

    colors_Oxy = ['#9ecae1']  # ['#deebf7','#9ecae1','#3182bd']
    colors_LARC = ['#fdae6b']  # ['#fee6ce','#fdae6b','#e6550d']
    colors_Comb = ['#a1d99b']  # ['#e5f5e0','#a1d99b','#31a354']

    uf.violinplot(dictionary['LARC-RRP'], 20, 20, '', colors_LARC)
    # catplot_aug(dictionary['LARC-RRP'], 20, 20, '', col_names, colors_LARC, save=False)

#main_aug()

def main_kfold(sheet_name, LARC=False):

    excel_path = '/Volumes/LaCie/MasterThesis_Ingvild/Excel_data/Experiment_plan.xlsx'
    dataframe = pd.read_excel(excel_path, sheet_name=sheet_name)
    paths = list(dataframe['Result path'])

    # Define correct experiments (IDs)
    if LARC:
        ids = [2, 1, 0, 3, 4]
        column_names = ['Fold 3', 'Fold 2', 'Fold 1', 'Fold 4', 'Fold 5']
    else:
        ids = [4, 1, 2, 3, 0]
        column_names = ['Fold 5', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 1']

    df = pd.DataFrame()

    for i in range(len(column_names)):
        df_temp = pd.read_csv(paths[ids[i]] + '/logs.csv')
        df[column_names[i]] = df_temp['val_dice']
        print(column_names[i])
        print('Max val_dice score:', max((list(df_temp['val_dice']))))

    max_dsc = []

    if LARC:
        df = uf.swap_columns(df, 'Fold 3', 'Fold 1')
        for i in list(df.columns):
            max_dsc.append(max(list(df[i])))
        df['Data'] = 'LARC-RRP'
        df = pd.melt(df, id_vars=['Data'], var_name=['Parameters'])
        colors = ['#fdae6b']
        uf.boxplot(df, 20, 20, '', colors)
    else:
        df = uf.swap_columns(df, 'Fold 5', 'Fold 1')
        for i in list(df.columns):
            max_dsc.append(max(list(df[i])))
        df['Data'] = 'Oxytarget'
        df = pd.melt(df, id_vars=['Data'], var_name=['Parameters'])
        colors = ['#9ecae1']
        uf.boxplot(df, 20, 20, '', colors)

    return max_dsc

def main_lr():
    excel_path = '/Volumes/LaCie/MasterThesis_Ingvild/Excel_data/Experiment_plan.xlsx'

    # Define correct experiments (IDs)
    ids_LARC = [5, 8]
    ids_Oxy = [5, 8]
    ids_Comb = [0, 3]

    # Creating dataframes of det f1 scores of the validation patients
    Oxy = uf.dataframe_of_f1scores(excel_path, 'Oxy_new', ['patient.csv'], ids_Oxy)
    LARC = uf.dataframe_of_f1scores(excel_path, 'LARC', ['patient_352.csv', 'patient_256.csv'], ids_LARC)
    Combined = uf.dataframe_of_f1scores(excel_path, 'Combined_new', ['patient_352.csv', 'patient_256.csv'], ids_Comb)

    # Creating dictionary of dataframes
    dictionary = {'OxyTarget': Oxy, 'LARC-RRP': LARC, 'Combined': Combined}
    medians = {}

    col_names_lr = ['1e-03', '1e-04', '1e-05']
    col_names_loss = ['Dice', 'Modified Dice']

    for key in dictionary:
        print(key)
        #dictionary[key] = uf.swap_columns(dictionary[key], '1e-04+Dice', '1e-03+Dice')
        for i in range(len(dictionary[key].columns)):
            dictionary[key].columns.values[i] = col_names_loss[i]
            if not col_names_loss[i] in medians:
                medians[col_names_loss[i]] = [dictionary[key].median()[i]]
            else:
                medians[col_names_loss[i]].append(list(dictionary[key].median())[i])

    colors_Oxy = '#9ecae1' # ['#deebf7','#9ecae1','#3182bd']
    colors_LARC = '#fdae6b'  # ['#fee6ce','#fdae6b','#e6550d']
    colors_Comb = '#a1d99b'  # ['#e5f5e0','#a1d99b','#31a354']

    colors = [colors_Oxy, colors_LARC, colors_Comb]
    markers = ['o', 's', '^']
    print(medians)
    print(np.mean(medians['Dice']))
    print(np.mean(medians['Modified Dice']))

    #uf.plot_learning_rates(medians, ['OxyTarget', 'LARC-RRP', 'Combined'], colors, markers)



main_lr()



