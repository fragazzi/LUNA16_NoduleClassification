# DEEP FEATURES FUSION WITH CONCATENATION
# Second technique proposed in the paper

import numpy as np

from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score

import torch
from torch.utils.data import DataLoader

from Utils import load_nodules
from Dataset import SlicesDataset
from Models import *
from MyConfig import *

# Get the gpu device by its IDs
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} as device\n')

# Feature extractors instantiation
vgg16 = VGGExtractor('VGG16')
vgg19 = VGGExtractor('VGG19')
alexnet = AlexNetExtractor()

vgg16 = vgg16.to(device)
vgg19 = vgg19.to(device)
alexnet = alexnet.to(device)
print(f'Feature extractors created')
print('-' * 25)

# Classifier (SVM) instantiation with GridSearch
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['linear', 'rbf']
    }
clf = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=3)
print(f'One classifier created:\n{clf}')
print('-'*25)
"""
# Classifier (Adaboost) instantiation with GridSearch
param_grid = {
    'n_estimators': [50, 100, 200, 500],
    'learning_rate': [5, 1, 0.1, 0.01, 0.001, 0.0001],
    'algorithm': ['SAMME', 'SAMME.R']
    }
clf = GridSearchCV(AdaBoostClassifier(), param_grid, refit=True, verbose=3)
print(f'One classifier created:\n{clf}')
print('-'*25)
"""

# TRAINING
print('> TRAINING PHASE')

# Load nodules ROIs and labels
train_rois, train_labels = load_nodules(TRAIN_DIR)
print('Loaded train nodules')
print(f'ROIs: {train_rois.shape}, labels: {train_labels.shape}')

# Create dataset and dataloader
train_dataset = SlicesDataset(train_rois, train_labels, transform=True)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

# Training features extraction and concatenation loop
print('Features extraction and concatenation loop...', end=' ')
train_features_list = []
for image, label in train_dataloader:
    image = torch.Tensor(image.numpy().astype(np.double)).to(device)

    alexnet_features = alexnet(image).squeeze().cpu().detach().numpy()
    vgg16_features = vgg16(image).squeeze().cpu().detach().numpy()
    vgg19_features = vgg19(image).squeeze().cpu().detach().numpy()

    concatenated_features = np.concatenate([alexnet_features, vgg16_features, vgg19_features])
    train_features_list.append(concatenated_features)

train_features_array = np.array(train_features_list)
print('finished')
print(f'Training features array shape: {train_features_array.shape}')

# Training of the classifier
clf.fit(train_features_array, train_labels)
print('\nClassifier trained')
print('-'*25)


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
print('\nStarting test nodules classification loop...', end=' ')

test_features_list = []
y_true, y_pred = [], []
for i, (image, label) in enumerate(test_dataloader):
    image = torch.Tensor(image.numpy().astype(np.double)).to(device)

    alexnet_features = alexnet(image).squeeze().cpu().detach().numpy()
    vgg16_features = vgg16(image).squeeze().cpu().detach().numpy()
    vgg19_features = vgg19(image).squeeze().cpu().detach().numpy()

    concatenated_features = np.concatenate([alexnet_features, vgg16_features, vgg19_features])
    test_features_list.append(concatenated_features)

    # Compute nodule prediction with MAX VOTING of slices prediction
    if (i + 1) % N_SLICES == 0:
        test_features_array = np.array(test_features_list)
        xSlice_predictions = clf.predict(test_features_array)

        nodule_prediction = np.bincount(xSlice_predictions).argmax()

        y_pred.append(nodule_prediction)
        y_true.append(test_labels[i])

        test_features_list = []

print('finished')
print('-' * 25)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)
print('CONFUSION MATRIX xNodules:')
print(cm)
print('\n')

# Compute precision and recall
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
print(f'Precision: {precision}\nRecall: {recall}')

# Number of nodule well predicted
print(f'> Correct nodules: {(np.array(y_true) == np.array(y_pred)).sum()}')

print('END')
