# Script to extract nodules voxel from the CT scan of LUNA16 

import os
import glob

import pandas as pd
import numpy as np

import SimpleITK as sitk

from tqdm import tqdm

from Config import *

# Number of slices of a final nodule voxel
# Change this number if you want more slices for each nodule
N_SLICES = 3

# Read the annotations file
df = pd.read_csv(ANNOTATIONS_DIR + '/full_annotations.csv', sep=';').drop(columns=['Unnamed: 0'])
print('Annotations file read')

# Susbet folders list
subset_list = os.listdir(SUBSETS_DIR)

# Main loop
nodule_number = 0
nodule_id_list, nodule_name_list = [], []

for subset in subset_list:
    print(f'Processing {subset}:')
    scans_path_list = glob.glob(SUBSETS_DIR + '/' + subset + '/'  + "*.mhd")
    
    for scan_p in tqdm(scans_path_list):
        image = sitk.ReadImage(scan_p)
        image_array = sitk.GetArrayFromImage(image)
        
        scan_id = scan_p.split('/')[-1][:-4]
        scan_df = df[df.seriesuid == scan_id]
        
        if len(scan_df) > 0: # discard scans without annotations
            coords_list = []
            for index, row in scan_df.iterrows():    
                coords_dict = {
                    'uid': row['seriesuid'],
                    'x': row['coordX'],
                    'y': row['coordY'],
                    'z': row['coordZ'],
                    'diameter': row['diameter_mm'],
                    'class': row['class']
                }                
                coords_list.append(coords_dict)
        
            for idx in range(len(coords_list)):   
                current_nodule = coords_list[idx]
                
                # World coordinates to voxel coordinates conversion
                x_w, y_w, z_w = current_nodule['x'], current_nodule['y'], current_nodule['z'] 
                x_v, y_v, z_v = image.TransformPhysicalPointToIndex((x_w, y_w, z_w))
                
                uid = current_nodule['uid']
                label = current_nodule['class']
                nodule_voxel = []

                idx_range = (N_SLICES - 1) // 2
                
                for i in reversed(range(1, idx_range+1)):
                    slice_before = image_array[z_v - i]
                    nodule_voxel.append(slice_before[y_v-32:y_v+32, x_v-32:x_v+32])
                
                slice_current = image_array[z_v]
                nodule_voxel.append(slice_current[y_v-32:y_v+32, x_v-32:x_v+32])
                
                for i in range(1, idx_range+1):
                    slice_after = image_array[z_v + 1]
                    nodule_voxel.append(slice_after[y_v-32:y_v+32, x_v-32:x_v+32])
                    
                nodule_voxel = np.array(nodule_voxel)
                
                # Create a SimpleITK image from the NumPy array
                nodule_voxel_sitk = sitk.GetImageFromArray(nodule_voxel)

                # Set the image spacing and origin
                nodule_voxel_sitk.SetSpacing([1, 1, 1])
                nodule_voxel_sitk.SetOrigin([0, 0, 0])

                # Save the image as NIFTI format
                try:
                    nodule_number += 1
                    nodule_name = 'LUNA16_N_%04d_L_%01d' % (nodule_number, label)
                    
                    sitk.WriteImage(nodule_voxel_sitk, DATASET_DIR + f'/final_volumes/{nodule_name}.nii')
                    
                    # If the nodule voxel is saved store original id and new nodule_name
                    nodule_id_list.append(uid)
                    nodule_name_list.append(nodule_name)
                    
                except Exception as e:
                    print('> Problem with one nodule')
                    print(f'> Id: {uid}')
                    print(f'> Error: {e}')
                    
print('All subsets processed')

# Create dataframe with uid and names (could be helpful?)
data_dict = {
    'seriesuid': nodule_id_list,
    'nodule_name': nodule_name_list
}

name_df = pd.DataFrame(data_dict)
name_df.to_csv(DATASET_DIR + '/nodule_names.csv')
print('nodule_names df saved')

print('END')