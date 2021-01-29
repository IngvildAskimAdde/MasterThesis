
import os
import pandas as pd
import SimpleITK as sitk
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def dataset_dataframe(path_main_folder):

    df = pd.DataFrame(columns=['File','Path']) #TODO: Add PatientID in dataframe
    for root, dirs, files in os.walk(path_main_folder):
        if root.endswith('MRS1'):
            paths = [os.path.join(root, filename) for filename in files]

        else:
            paths = [os.path.join(root, filename) for filename in files]

        df1 = pd.DataFrame({'File': files, 'Path': paths})
        df = df.append(df1)

    #df['PatientID'] = df['Path'].map(lambda x: x.lstrip('/'))

    df = df.sort_index()
    df = df[~df['File'].astype(str).str.startswith('._')] #Removes filenames starting with ._ (due to copying of files)
    df['RowNumber'] = list(range(0,df.shape[0]))
    df = df.set_index('RowNumber')

    return df



df = dataset_dataframe('/Volumes/Untitled 1/LARC_T2_cleaned_nii')
#df = dimensions('/Volumes/Untitled 1/Ingvild_Oxytarget')

def dimensions(dataset_dataframe):
    dataset_dataframe['xDimension'] = ''
    dataset_dataframe['yDimension'] = ''
    dataset_dataframe['zDimension'] = ''
    for i, row in dataset_dataframe.iterrows():
        image = sitk.ReadImage(row[1])
        array = sitk.GetArrayFromImage(image)
        dim = np.shape(array)
        print(dim[1], dim[2], dim[0])
        dataset_dataframe['xDimension'][i] = dim[1]
        dataset_dataframe['yDimension'][i] = dim[2]
        dataset_dataframe['zDimension'][i] = dim[0]
    return dataset_dataframe

df = dimensions(df)
df_imgFile = df.iloc[:int(df.shape[0]/3)]
df_maskFile = df.iloc[2*int(df.shape[0]/3):]

print('Image files: ', df_imgFile['xDimension'].value_counts())
print('Mask files: ', df_maskFile['xDimension'].value_counts())

plt.figure()
sns.histplot(df_imgFile, x='xDimension')
plt.title('(256,256)=6, (384,384)=1, (512,512)=49, (640,640)=33')

plt.figure()
sns.histplot(df_maskFile, x='xDimension')