import pickle
import glob
import os
import numpy as np
import pandas as pd
from nilearn import plotting, image
import nibabel as nib
import joblib
from joblib import Parallel, delayed



 
NTFA_path = "/work/abslab/NTFA_packages/NTFADegeneracy_merged/"
print("NTFA code from:", NTFA_path)
import sys
sys.path.append(NTFA_path)
import htfa_torch.utils as utils


query_dir = 'models/AVFP_NTFA_sub-All_epoch-2300_factor-100_mask-GMgroup_111_lin-None_ntfa-v2_visreg/'

def single_permutation(file_list, i, query_dir):
    perm_corr_list = []
    shuffled_file_list = np.random.RandomState(seed=i).permutation(file_list)

    for act_file, recon_file in zip(file_list, shuffled_file_list):
        act_path = query_dir + 'activations/' + act_file + '.nii.gz'
        recon_path = query_dir + 'reconstruction/' + recon_file + '.nii.gz'

        activations = utils.nii2cmu(act_path)
        recontructions = utils.nii2cmu(recon_path)

        corr = np.round(np.corrcoef(activations['data'][:, 0], recontructions['data'][:, 0])[0, 1], 3)
        perm_corr_list.append(corr)
    
    np.save(query_dir + f'recon_corr_permut/recon_corr_all_permutations_{str(i)}.npy', np.array(perm_corr_list))
    return perm_corr_list

def parallel_permutation_test(file_list, query_dir, n_permutations=100, n_jobs=-1):
    all_permutations = Parallel(n_jobs=n_jobs)(
        delayed(single_permutation)(file_list, i, query_dir) for i in range(n_permutations)
    )

    # Combine the results into a single numpy array
    all_permutations_array = np.array(all_permutations)
    return all_permutations_array


file_list = joblib.load('recon_correlation_file_order_list.joblib')
# Perform the parallel permutation test
n_permut = 1000
all_permutations_array = parallel_permutation_test(file_list,query_dir, n_permut)

# Save the array to a file
np.save(query_dir + f'recon_corr_all_permutations_{n_permut}_do_not_change.npy', all_permutations_array)