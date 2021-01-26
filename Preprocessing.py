
import os
import pandas as pd

def dimensions(path_main_folder):

    df = pd.DataFrame(columns=['File','Path']) #TODO: Add PatientID in dataframe
    for root, dirs, files in os.walk(path_main_folder):

        print(dirs)

        if root.endswith('MRS1'):
            paths = [os.path.join(root, filename) for filename in files]

        else:
            paths = [os.path.join(root, filename) for filename in files]

        df1 = pd.DataFrame({'File': files, 'Path': paths})
        df = df.append(df1)

        #TODO: Group File column in dataframe

    return df



df = dimensions('/Volumes/Untitled 1/LARC_T2_preprocessed')
#df = dimensions('/Volumes/Untitled 1/Ingvild_Oxytarget')

