import os

import numpy as np

import SimpleITK as sitk

from tqdm import tqdm


def load_nodules(folder_dir):
    """
    Load nodules ROIs from a folder and return two numpy array:
    - X: numpy array with all the slices
    - y: numpy array with labels

    Args:
        folder_dir (str): path to folder with nodules ROIs
    """

    nodules_list = os.listdir(folder_dir)
    x, y = [], []

    for name in tqdm(nodules_list):
        image = sitk.ReadImage(folder_dir + '/' + name)
        image_array = sitk.GetArrayFromImage(image)

        label = int(name.split('_')[6].split('.')[0])

        for idx in range(len(image_array)):
            x.append(image_array[idx])
            y.append(label)

    return np.array(x), np.array(y)
