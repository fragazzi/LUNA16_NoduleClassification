# FILE PATHS

HOMEDATA_DIR = '/mnt/c/Users/checc/MasterThesis/LUNA16'

DATASET_DIR = HOMEDATA_DIR + '/dataset'

SUBSETS_DIR = DATASET_DIR + '/volumes/images'

FINAL_DATASET_DIR = DATASET_DIR + '/final_volumes'

TRAIN_DIR = DATASET_DIR + '/train'

TEST_DIR = DATASET_DIR + '/test'

ANNOTATIONS_DIR = '/mnt/c/Users/checc/MasterThesis/LUNA16/annotations'


# CONSTANT VALUES

# Number of slices of a final nodule voxel
# Change this number if you want more slices for each nodule
N_SLICES = 3

# Number of nodules for test
N_TEST_NODULES = 200


# MODEL NAME
# Deep neural network used to extract deep features
MODEL_NAME = 'AlexNet'