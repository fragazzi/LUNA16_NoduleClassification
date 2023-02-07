# Second test:
# Features extraction using a pre-trained model and classification with a ML model
# MAX VOTING for nodules classification
# Slice repetition on the z-axis

import re

import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.metrics import confusion_matrix

import torch
from torch.utils.data import DataLoader

from Utils import load_nodules
from Dataset import SlicesDataset
from Models import *
from Config import *


# Get the gpu device by its IDs
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} as device\n')

# Feature extractor instantiation
if re.split('(\d+)', MODEL_NAME)[0] == 'VGG':
    fe = VGGExtractor(MODEL_NAME)
    
elif MODEL_NAME == 'AlexNet':
    fe = AlexNetExtractor()

fe = fe.to(device)
print(f'Feature extrator created:\n{MODEL_NAME}')
print('-'*15)

# Classifier (SVM) instantiation
C = 1
kernel = 'linear'
gamma = 'scale'
clf = svm.SVC(C=C, kernel=kernel, gamma=gamma)
print(f'Classifier created:\n{clf}')
print('-'*15)


# TRAINING
print('> TRAINING PHASE')

# Load nodules ROIs and labels
train_rois, train_labels = load_nodules(TRAIN_DIR)
print('Loaded train nodules')
print(f'ROIs: {train_rois.shape}, labels: {train_labels.shape}')

# Create dataset and dataloader
trin_dataset = SlicesDataset(train_rois, train_labels, transform=True)
train_dataloader = DataLoader(trin_dataset, batch_size=1, shuffle=False)

# Training features extraction loop
print('\nStarting training extraction loop...')

train_features_list = []
for (image, label) in train_dataloader:
    image = torch.Tensor(image.numpy().astype(np.double)).to(device)
    
    features = fe(image)
    train_features_list.append(features.squeeze().cpu().detach().numpy())
    
train_features_array = np.array(train_features_list)

print('Extraction loop finished')
print(f'Training features array shape: {train_features_array.shape}')

# Training of the classifier
clf.fit(train_features_array, train_labels)
print('\nClassifier trained')
print('-'*15)


# TESTING PHASE
print('> TESTING PHASE')

# Load nodules ROIs and labels
test_rois, test_labels = load_nodules(TEST_DIR)
print('Loaded train nodules')
print(f'ROIs: {test_rois.shape}, labels: {test_labels.shape}')

# Create dataset and dataloader
test_dataset = SlicesDataset(test_rois, test_labels, transform=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Testing features extraction loop
print('\nStarting test noudles classification loop...')

test_features_list = []
y_true, y_pred = [], []
for i, (image, label) in enumerate(test_dataloader):
    image = torch.Tensor(image.numpy().astype(np.double)).to(device)
    
    features = fe(image)
    test_features_list.append(features.squeeze().cpu().detach().numpy())
    
    # Compute nodule prediction with MAX VOTING of slices prediction
    if (i+1) % N_SLICES == 0:
        test_features_array = np.array(test_features_list)
        xSlice_predictions = clf.predict(test_features_array)
        
        nodule_prediction = np.bincount(xSlice_predictions).argmax()
        
        y_pred.append(nodule_prediction)
        y_true.append(test_labels[i])   
        
        test_features_list = []         
             
print('Classification loop finished')
print('-'*25)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)
print('CONFUSION MATRIX xNodules:')
print(cm)
print('-'*25)

# Save confusione matrix as csv file
cm_df = pd.DataFrame(cm, columns=['Predicted Positive', 'Predicted Negative'], index=['Actual Positive', 'Actual Negative'])
cm_df.to_csv('./results/confusion_matrix.csv', index=False)
print('Confusion matrix saved')

print('\nEND')