
import useful_functions as uf
import ImageViewer as iv
import SimpleITK as sitk
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import second_delineation as sd


#path1 = '/Volumes/LaCie/MasterThesis_Ingvild/Experiments/LARC/LARC_ID_53/prediction.039.h5'
#path2 = '/Volumes/LaCie/MasterThesis_Ingvild/HDF5_data/traditionalSplit_Oxy_MatchedHistZScore_twoMasks.h5'
#uf.plot_image_slice(path1, indice=4)

#image_path = '/Volumes/LaCie/MasterThesis_Ingvild/Data/LARC_cropped/LARC-RRP-022/image.nii'
#mask_path_1 = '/Volumes/LaCie/MasterThesis_Ingvild/Data/LARC_cropped/LARC-RRP-022/1 RTSTRUCT LARC_MRS1-label.nii'

#path1 = '/Volumes/LaCie/MasterThesis_Ingvild/Experiments/LARC/LARC_ID_43/prediction.044.h5'
#path2 = '/Volumes/LaCie/MasterThesis_Ingvild/HDF5_data/traditionalSplit_LARC_352_MHZScore.h5'
#uf.plot_image_slice(path1, indice=350)

#image_path_1 = '/Volumes/Untitled/LARC_T2_cleaned_nii/LARC-RRP-075/MRS1/image.nii'
image_path_1 = '/Volumes/LaCie/MasterThesis_Ingvild/Data/LARC/LARC_cropped/LARC-RRP-075/image.nii'
#mask_path_1 = '/Volumes/Untitled/LARC_T2_cleaned_nii/LARC-RRP-075/MRS1/1 RTSTRUCT LARC_MRS1-label.nii'
mask_path_1 = '/Volumes/LaCie/MasterThesis_Ingvild/Data/LARC/LARC_cropped/LARC-RRP-075/1 RTSTRUCT LARC_MRS1-label.nii'
#mask_path_2 = '/Volumes/LaCie/MasterThesis_Ingvild/Data/Oxy_allData_MatchedHistZScore/Oxytarget_43_PRE/Manual_shh.nii'
uf.plot_slice_nifti(image_path_1, 20, mask_path_1)


image_1 = sitk.ReadImage(image_path_1)
mask_1 = sitk.ReadImage(mask_path_1)
#mask_2 = sitk.ReadImage(mask_path_2)

v = iv.Viewer(view_mode='2', mask_to_show=['a','b'])
v.set_image(image_1, label='image')
v.set_mask(mask_1, label='mask 1', color_rgb=[60, 180, 75])
#v.set_mask(mask_2, label='mask 2')
v.show()

"""
image_path_2 = '/Volumes/LaCie/MasterThesis_Ingvild/Data/dwi/Oxy_all_cropped_TS/Oxytarget_97_PRE/b4.nii'
mask_path_2 = '/Volumes/LaCie/MasterThesis_Ingvild/Data/dwi/Oxy_all_cropped_TS/Oxytarget_97_PRE/Manual_an.nii'
#mask_path_2 = '/Volumes/LaCie/MasterThesis_Ingvild/Data/Oxy_allData_MatchedHistZScore/Oxytarget_43_PRE/Manual_shh.nii'

image_2 = sitk.ReadImage(image_path_2)
mask_2 = sitk.ReadImage(mask_path_2)
#mask_2 = sitk.ReadImage(mask_path_2)

v = iv.Viewer(view_mode='2', mask_to_show=['a','b'])
v.set_image(image_2, label='image')
v.set_mask(mask_2, label='mask 1', color_rgb=[60, 180, 75])
#v.set_mask(mask_2, label='mask 2')
v.show()



uf.show_image_interactive('/Volumes/Untitled 1/Ingvild_Oxytarget/Oxytarget_74_PRE/T2.nii',
                         '/Volumes/Untitled 1/Ingvild_Oxytarget/Oxytarget_74_PRE/Manual_an.nii',
                          '1')

uf.show_image('/Volumes/Untitled 1/Ingvild_Oxytarget/Oxytarget_74_PRE/T2.nii',
              '/Volumes/Untitled 1/Ingvild_Oxytarget/Oxytarget_74_PRE/Manual_an.nii',
              6)

path1 = '/Volumes/LaCie/MasterThesis_Ingvild/Experiments/Oxy_new/Oxy_ID_24_new/prediction.092.h5'
path2 = '/Volumes/LaCie/MasterThesis_Ingvild/HDF5_data/traditionalSplit_Oxy_MatchedHistZScore_twoMasks.h5'
indice = 10
file1 = h5py.File(path1,'r')
file2 = h5py.File(path2,'r')
#data = file['val/352/input'][indice]
data = file1['x'][indice]
#mask = file['val/352/target_an'][indice]
mask1 = file1['y'][indice]
mask2 = file2['val/352/target_an'][indice]
predicted = file1['predicted'][indice]
#patient = file['val/352/patient_ids'][indice]
#print(data)
#print(data.shape)
#print(data[0][...,0].shape)
#print(data.max())
#print(mask.max())
#print(patient)

legend_elements = [Line2D([0], [0], color='red', label='Predicted Mask', lw=4),
                 Line2D([0], [0], color='gold', label='Mask 1', lw=4),
                   Line2D([0], [0], color='turquoise', label='Mask 2', lw=4)]

plt.figure(figsize=(12,8))
plt.imshow(data, cmap='gray')#, vmin=-2, vmax=4)
plt.colorbar()
plt.contourf(predicted[...,0], levels=[0.5,1.0], alpha=0.2, colors='red')
plt.contour(predicted[...,0], levels=[0.5], linewidths=2.5, colors='red')
plt.contourf(mask1[...,0], levels=[0.5,1.0], alpha=0.2, colors='gold')
plt.contour(mask1[...,0], levels=[0.5], linewidths=2.5, colors='gold')
plt.contourf(mask2[...,0], levels=[0.5,1.0], alpha=0.2, colors='turquoise')
plt.contour(mask2[...,0], levels=[0.5], linewidths=2.5, colors='turquoise')
plt.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, -0.15), ncol=3)
plt.axis('off')
plt.tight_layout()
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



def main_aug():

    excel_path = '/Volumes/LaCie/MasterThesis_Ingvild/Excel_data/Experiment_plan.xlsx'

    # Define correct experiments (IDs)
    ids_LARC = [8, 32, 33, 34]
    ids_Oxy = [8, 21, 22, 23]
    ids_Comb = [3, 14, 13, 15]

    # Creating dataframes of det f1 scores of the validation patients
    Oxy = uf.dataframe_of_f1scores(excel_path, 'Oxy_new', ['patient.csv'], ids_Oxy)
    LARC = uf.dataframe_of_f1scores(excel_path, 'LARC', ['patient_352.csv', 'patient_256.csv'], ids_LARC)
    Combined = uf.dataframe_of_f1scores(excel_path, 'Combined_new', ['patient_352.csv', 'patient_256.csv'], ids_Comb)

    # Creating dictionary of dataframes
    dictionary = {'OxyTarget': Oxy, 'LARC-RRP': LARC, 'Combined': Combined}

    #col_names_aug = ['No', 'Default', 'Best Combination']
    #col_names_aug = ['No', 'Default', 'BC']
    #col_names_norm = ['No', 'Z-Score', 'Matched Hist', 'Matched Hist + Z-Score']
    col_names_norm = ['No', 'Z-Score', 'MH', 'MH + Z-Score']


    for key in dictionary:
        print(key)
        for i in range(len(dictionary[key].columns)):
            dictionary[key].columns.values[i] = col_names_norm[i]
            print(dictionary[key].median())
        dictionary[key]['Data'] = key
        dictionary[key] = pd.melt(dictionary[key], id_vars=['Data'], var_name=['Parameters'])

    colors_Oxy = ['#9ecae1']  # ['#deebf7','#9ecae1','#3182bd']
    colors_LARC = ['#fdae6b']  # ['#fee6ce','#fdae6b','#e6550d']
    colors_Comb = ['#a1d99b']  # ['#e5f5e0','#a1d99b','#31a354']

    uf.violinplot(dictionary['OxyTarget'], 20, 20, '', colors_Oxy)
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


#main_kfold('LARC', True)


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
    #markers = ['o', 's', '^']
    markers = ['X', 'D']
    print(medians)
    print(np.mean(medians['Dice']))
    print(np.mean(medians['Modified Dice']))

    uf.plot_loss_functions(medians, ['OxyTarget', 'LARC-RRP', 'Combined'], colors, markers)

#main_lr()


def main_val():

    # Define correct experiments csv files
    Oxy_ID_27_352 = pd.read_csv('/Volumes/LaCie/MasterThesis_Ingvild/Experiments/Oxy_new/Oxy_ID_27_new/patient_352.csv')
    Oxy_ID_27_256 = pd.read_csv('/Volumes/LaCie/MasterThesis_Ingvild/Experiments/Oxy_new/Oxy_ID_27_new/patient_256.csv')
    Oxy_ID_27 = Oxy_ID_27_352.append(Oxy_ID_27_256, ignore_index=True)
    Oxy_ID_27 = Oxy_ID_27.drop(['patient_ids'], axis=1)
    Oxy_ID_27.rename(columns={'f1_score': 'Trained on OxyTarget'}, inplace=True)
    Oxy_ID_27['Data'] = 'OxyTarget'
    print(Oxy_ID_27.median())

    Oxy_ID_24 = pd.read_csv('/Volumes/LaCie/MasterThesis_Ingvild/Experiments/Oxy_new/Oxy_ID_24_new/patient.csv')
    Oxy_ID_24 = Oxy_ID_24.drop(['patient_ids'], axis=1)
    Oxy_ID_24.rename(columns={'f1_score': 'Trained on OxyTarget'}, inplace=True)
    Oxy_ID_24['Data'] = 'OxyTarget'
    print(Oxy_ID_24.median())

    LARC_ID_38 = pd.read_csv('/Volumes/LaCie/MasterThesis_Ingvild/Experiments/LARC/LARC_ID_38/patient.csv')
    LARC_ID_38 = LARC_ID_38.drop(['patient_ids'], axis=1)
    LARC_ID_38.rename(columns={'f1_score': 'Trained on LARC-RRP'}, inplace=True)
    LARC_ID_38['Data'] = 'LARC-RRP'
    print(LARC_ID_38.median())

    LARC_ID_35_352 = pd.read_csv('/Volumes/LaCie/MasterThesis_Ingvild/Experiments/LARC/LARC_ID_35/patient_352.csv')
    LARC_ID_35_256 = pd.read_csv('/Volumes/LaCie/MasterThesis_Ingvild/Experiments/LARC/LARC_ID_35/patient_256.csv')
    LARC_ID_35 = LARC_ID_35_352.append(LARC_ID_35_256, ignore_index=True)
    LARC_ID_35 = LARC_ID_35.drop(['patient_ids'], axis=1)
    LARC_ID_35.rename(columns={'f1_score': 'Trained on LARC-RRP'}, inplace=True)
    LARC_ID_35['Data'] = 'LARC-RRP'
    print(LARC_ID_35.median())

    #df = pd.concat([LARC_ID_35, Oxy_ID_27])
    df = pd.concat([Oxy_ID_24, LARC_ID_38])
    df = pd.melt(df, id_vars=['Data'], var_name=['Parameters'])
    #print(df)

    colors_LARC = ['#fdae6b']  # ['#fee6ce','#fdae6b','#e6550d']
    colors_Oxy = ['#9ecae1']  # ['#deebf7','#9ecae1','#3182bd']
    uf.violinplot(df, 20, 20, '', colors_Oxy)

#main_val()

def main_valfolds_1():
    Oxy_ID_27_352 = pd.read_csv('/Volumes/LaCie/MasterThesis_Ingvild/Experiments/Oxy_new/Oxy_ID_27_new/patient_352.csv')
    Oxy_ID_27_256 = pd.read_csv('/Volumes/LaCie/MasterThesis_Ingvild/Experiments/Oxy_new/Oxy_ID_27_new/patient_256.csv')

    LARC_ID_35_352 = pd.read_csv('/Volumes/LaCie/MasterThesis_Ingvild/Experiments/LARC/LARC_ID_35/patient_352.csv')
    LARC_ID_35_256 = pd.read_csv('/Volumes/LaCie/MasterThesis_Ingvild/Experiments/LARC/LARC_ID_35/patient_256.csv')

    legend_elements = [Line2D([0], [0], marker='o', color='k', label='Trained on OxyTarget data',
                              markerfacecolor='none', markersize=15, linestyle='none'),
                       Line2D([0], [0], marker='^', color='k', label='Trained on LARC-RRP data',
                              markerfacecolor='none', markersize=15, linestyle='none'),
                       Line2D([0], [0], color='#fdae6b', label='Image dimension of 352x352', lw=4),
                       Line2D([0], [0], color='#fe420f', label='Image dimension of 256x256', lw=4), ]

    plt.figure(figsize=(14,8))
    plt.scatter(Oxy_ID_27_352['patient_ids'], Oxy_ID_27_352['f1_score'], marker='o', color='#fdae6b', facecolors='none',
                s=200, linewidths=2)
    plt.scatter(Oxy_ID_27_256['patient_ids'], Oxy_ID_27_256['f1_score'], marker='o', color='#fe420f', facecolors='none',
                s=200, linewidths=2)
    plt.scatter(LARC_ID_35_352['patient_ids'], LARC_ID_35_352['f1_score'], marker='^', color='#fdae6b',
                facecolors='none', s=200, linewidths=2)
    plt.scatter(LARC_ID_35_256['patient_ids'], LARC_ID_35_256['f1_score'], marker='^', color='#fe420f',
                facecolors='none', s=200, linewidths=2)
    plt.ylabel(r'DSC$_{\mathrm{P}}$')
    plt.xlabel('Patient IDs')
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.show()

#main_valfolds_1()

def main_valfolds_2():
    Oxy_ID_24 = pd.read_csv('/Volumes/LaCie/MasterThesis_Ingvild/Experiments/Oxy_new/Oxy_ID_24_new/patient.csv')
    LARC_ID_38 = pd.read_csv('/Volumes/LaCie/MasterThesis_Ingvild/Experiments/LARC/LARC_ID_38/patient.csv')

    legend_elements = [Line2D([0], [0], marker='o', color='k', label='Trained on OxyTarget data',
                              markerfacecolor='none', markersize=15, linestyle='none'),
                       Line2D([0], [0], marker='^', color='k', label='Trained on LARC-RRP data',
                              markerfacecolor='none', markersize=15, linestyle='none'),
                       Line2D([0], [0], color='#9ecae1', label='Image dimension of 352x352', lw=4),
                       ]

    plt.figure(figsize=(14, 8))
    plt.scatter(Oxy_ID_24['patient_ids'], Oxy_ID_24['f1_score'], marker='o', color='#9ecae1', facecolors='none', s=200,
                linewidths=2)
    plt.scatter(LARC_ID_38['patient_ids'], LARC_ID_38['f1_score'], marker='^', color='#9ecae1', facecolors='none',
                s=200, linewidths=2)
    plt.ylabel(r'DSC$_{\mathrm{P}}$')
    plt.xlabel('Patient IDs')
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.show()

#main_valfolds_2()
"""

#path_LARC_org = '/Volumes/LaCie/MasterThesis_Ingvild/Data/LARC_cropped/LARC-RRP-035/image.nii'
#path_Oxy = '/Volumes/LaCie/MasterThesis_Ingvild/Data/Oxy_cropped_corrected/Oxytarget_120_PRE/T2.nii'
#path_LARC_new = '/Volumes/LaCie/MasterThesis_Ingvild/Data/LARC_cropped_MatchedHistOnOxy/LARC-RRP-035/image.nii'
                      # 20, 2)
#uf.plot_slice_nifti(path_LARC_new, 20)

#path1 = '/Volumes/LaCie/MasterThesis_Ingvild/Experiments/Oxy_new/Oxy_ID_28_new/prediction.041.h5'
#path2 = '/Volumes/LaCie/MasterThesis_Ingvild/HDF5_data/traditionalSplit_Oxy_MatchedHistZScore_twoMasks.h5'
#uf.plot_image_slice(path1, indice=259)

#df1, df2 = sd.correct_patients_csv('/Volumes/LaCie/MasterThesis_Ingvild/Experiments/Oxy_new/Oxy_ID_24_new/slice.csv','/Volumes/LaCie/MasterThesis_Ingvild/Experiments/Oxy_new/Oxy_ID_24_new/mask2/slice_mask2.csv')
#df_352 = pd.read_csv('/Volumes/LaCie/MasterThesis_Ingvild/Experiments/Combined_new/Combined_ID_16_new/slice_352.csv')
#df_256 = pd.read_csv('/Volumes/LaCie/MasterThesis_Ingvild/Experiments/Combined_new/Combined_ID_16_new/slice_256.csv')
#df = df_352.append(df_256, ignore_index=True)
#df = pd.read_csv('/Volumes/LaCie/MasterThesis_Ingvild/Experiments/Oxy_new/Oxy_ID_28_new/slice.csv')
#df = uf.max_and_min_dsc_score(df)





