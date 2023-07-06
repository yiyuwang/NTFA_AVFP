


import sys
import glob
import os
import joblib
import numpy as np
import pandas as pd
import random
import nibabel as nib


import statistics 
import itertools


# for parallel processing
from joblib import Parallel, delayed, cpu_count

#scikit learn
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

from math import sqrt


def get_video_category(vn):
    if vn in [1,3,5,8,10,44,2,4,9,12,39,48]:
        cat = 'Heights'
    elif vn in [13,20,21,22,49,60,16,24,50,51,58,59]:
        cat = 'Social'
    elif vn in [29,33,35,61,64,72,32,36,65,69,70,71]:
        cat = 'Spiders'
    else:
        raise Exception('no this video number!')
    return cat 


def GetXY(embedding_df, file_list, file_path, which_rating = 'fear'):
    # Prepare the data
    X, y = [], []
    for file_name in file_list:
        act_file = file_path + file_name + '.nii.gz'
        activations = masker.fit_transform(act_file).ravel()
        X.append(activations)
        
        stimulus = file_name.split("video-")[1]
        rating = embedding_df.loc[embedding_df['stimulus'] == stimulus][which_rating].values[0]
        y.append(rating)

    X = np.array(X)
    y = np.array(y)
  
    return X, y
       
    

def GetXY_bysub(data_ids, data_column, mask_img, embedding_df, which_test, which_rating = 'fear'):
    nifti_masker = NiftiMasker(mask_img=mask_img,standardize_confounds=False)
    y = []
    X = []
    
    if isinstance(data_ids, list):
        data_index_list = []
        for d in data_ids:
            data_index_list.append(np.where(embedding_df[data_column] == d)[0])
        data_index = [item for sublist in data_index_list for item in sublist]  
    elif data_ids == 'all':
        data_index = embedding_df.index.to_list()
    else:
        data_index = np.where(embedding_df[data_column] == data_ids)[0]
        

    for i in data_index:

        y.append(embedding_df.loc[i, which_rating])

        run = int(embedding_df.loc[i, 'run'])
        stimulus = embedding_df.loc[i, 'stimulus']
        sub = int(embedding_df.loc[i,'participant'])

        act_file = query_dir + f'{which_test}/sub-{sub}_run-{run}_video-{stimulus}.nii.gz'

        X.append(nifti_masker.fit_transform(act_file).ravel())
    
    X = np.array(X)
    y = np.array(y)
    return X, y



def TrainTestBrainDecoder(model, kf, data_ids, data_column, which_test, which_y, embedding_df, mask_img):
    test_acc_scores = []
    test_rmse_scores = []
    for i, (train_idx, test_idx) in enumerate(kf.split(data_ids)):
        print(f'---------------------- fold {i+1} ---------------------------')

        data_train = np.array(data_ids)[train_idx.astype(int)].tolist()
        data_test = np.array(data_ids)[test_idx.astype(int)].tolist()

        X_train, y_train = GetXY_bysub(data_train, data_column, mask_img, embedding_df, which_test)
        X_test, y_test = GetXY_bysub(data_test, data_column, mask_img, embedding_df, which_test)

        model.fit(X_train,y_train)

        y_pred = model.predict(X_test)
        
        test_acc = np.corrcoef(y_test,y_pred)[0,1]
        test_acc_scores.append(test_acc)
        
        rmse = sqrt(mean_squared_error(y_test, y_pred))
        test_rmse_scores.append(rmse)
 
    return test_acc_scores, test_rmse_scores


SEED = int(sys.argv[1])
print(f"CV seed: {SEED}")

# set up directory:
query_dir = 'models/AVFP_NTFA_sub-All_epoch-2300_factor-100_mask-GMgroup_111_lin-None_ntfa-v2_visreg/'



# subj id
base_dir = '/work/abslab/Yiyu/NTFA_AVFP/'
included_data = pd.read_csv(base_dir + 'fmri_info/included_avfp_subjects.csv', header=None)
subIDs = included_data[0].astype('int').tolist()

# mask
mask_dir = '/work/abslab/AVFP/NTFA/masks/'
mask_type = 'GMgroup'
if mask_type == 'GMgroup':
    mask_file = mask_dir +f'GM_fmriprep_novelgroup_mask_N71.nii.gz'
else:
    mask_file = mask_dir + 'gm_mask_icbm152_brain.nii.gz'

from nilearn.input_data import NiftiMasker
masker = NiftiMasker(mask_img=mask_file)


# embedding df
print('loading embedding_df')
embedding_df = pd.read_csv(query_dir + 'all_embedding_recon_rating_df.csv')
    

n_split = 6
data_column = 'participant'
data_ids = subIDs
data_id_group = None
which_test = 'activations'
which_y = 'fear'


kernel = 'rbf'
model = svm.SVR(kernel = kernel)
print('model : ', model)


kf = KFold(n_splits=n_split, shuffle=True, random_state = SEED)

acc_scores, rmse_scores = TrainTestBrainDecoder(model, kf, data_ids, data_column, which_test, which_y, embedding_df, mask_file)

np.save(query_dir + f'/BrainDecoder_{which_test}_permutation/whole-brain_{kernel}_acc_permutation_{SEED}.npy', acc_scores)
np.save(query_dir + f'/BrainDecoder_{which_test}_permutation/whole-brain_{kernel}_rmse_permutation_{SEED}.npy', rmse_scores)