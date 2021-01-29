
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


def dimensions(dataset_dataframe):

    dataset_dataframe['xDimension'] = ''
    dataset_dataframe['yDimension'] = ''
    dataset_dataframe['zDimension'] = ''

    dataset_dataframe['xVoxelDimension'] = ''
    dataset_dataframe['yVoxelDimension'] = ''
    dataset_dataframe['zVoxelDimension'] = ''

    for i, row in dataset_dataframe.iterrows():
        image = sitk.ReadImage(row[1])
        array = sitk.GetArrayFromImage(image)
        dim = np.shape(array)
        print(image.GetSpacing()[0])

        dataset_dataframe['xDimension'][i] = dim[1]
        dataset_dataframe['yDimension'][i] = dim[2]
        dataset_dataframe['zDimension'][i] = dim[0]

        dataset_dataframe['xVoxelDimension'][i] = image.GetSpacing()[0]
        dataset_dataframe['yVoxelDimension'][i] = image.GetSpacing()[1]
        dataset_dataframe['zVoxelDimension'][i] = image.GetSpacing()[2]

    return dataset_dataframe

#df = dataset_dataframe('/Volumes/Untitled/LARC_T2_preprocessed')
df = dataset_dataframe('/Volumes/Untitled 1/Oxytarget_preprocessed')

df = dimensions(df)
df_imgFile = df.iloc[:int(df.shape[0]/2)]
#df_maskFile = df.iloc[2*int(df.shape[0]/3):]

#print('Image files: ', df_imgFile['xDimension'].value_counts())
#print('Mask files: ', df_maskFile['xDimension'].value_counts())
#print('Image files (voxelsize): ', df_imgFile['xVoxelDimension'].value_counts())

print('Max x-dimension:', df['xDimension'].max())
print('Max y-dimension:', df['yDimension'].max())
print('Max z-dimension:', df['zDimension'].max())

"""
plt.figure()
sns.histplot(df_imgFile, x='xDimension')
plt.title('OxyTarget image dimensions \n (512,512)=107, (528,528)=2, (560,560)=1')

plt.figure()
sns.histplot(df_imgFile, x='xVoxelDimension')
plt.title('OxyTarget voxelsizes')
          #'\n (0.323,0.323)=1, (0.340,0.340)=1, (0.342,0.342)=1, (0.344, 0.344)=1 \n (0.3467,0.3467)=1, (0.3477,0.3477)=1, (0.3478,0.3478)=1, (0.351,0.351)=1 \n (0.352,0.352)=65 ')
"""