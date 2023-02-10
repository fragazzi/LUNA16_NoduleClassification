# Script to split nodules voxel in training and testing folders

import os
import shutil
import random

from tqdm import tqdm

from Config import *


file_list = os.listdir(FINAL_DATASET_DIR)

# cretate malignant and benign nodules name list
malignant_list, benign_list = [], []
for file in file_list:
    tokens = file.split('_')
    label = int(tokens[-1].split('.')[0])
    
    if label == 0:
        malignant_list.append(file)
    else:
        benign_list.append(file)

print(f'Malignant nodules: {len(malignant_list)} -> tot slices: {len(malignant_list) * N_SLICES}')
print(f'Benign nodules: {len(benign_list)} -> tot slices: {len(benign_list) * N_SLICES}') 


# select randomly nodules for test
test_malignant_list = random.sample(malignant_list, N_TEST_NODULES//2)
test_benign_list = random.sample(benign_list, N_TEST_NODULES//2)

test_nodules_list = test_benign_list + test_malignant_list

# create train list
train_malignant_list = [file_name for file_name in malignant_list if file_name not in test_malignant_list]

train_benign_list = [file_name for file_name in benign_list if file_name not in test_benign_list]
train_benign_list = random.sample(train_benign_list, len(train_malignant_list))

train_nodules_list = train_benign_list + train_malignant_list

print('-'*15)
print(f'Number of train nodules: {len(train_nodules_list)} -> test dataset is balanced')
print(f'Number of test nodules: {len(test_nodules_list)} -> train dataset is balanced')


# save train and test nodules in the corresponding folders
print('-'*15)
print(f'Saving training nodules in {TRAIN_DIR}...')
for file in tqdm(train_nodules_list):
    src_path = os.path.join(FINAL_DATASET_DIR, file)
    dst_path = os.path.join(TRAIN_DIR, file)
    
    shutil.copy(src_path, dst_path)

print(f'Saving training nodules in {TEST_DIR}...')    
for file in tqdm(test_nodules_list):
    src_path = os.path.join(FINAL_DATASET_DIR, file)
    dst_path = os.path.join(TEST_DIR, file)
    
    shutil.copy(src_path, dst_path)

print('Done!')