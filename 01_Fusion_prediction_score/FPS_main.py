# FUSION BASED ON AVERAGING OF PREDICTION SCORE
# First technique proposed in the paper

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
clf_1 = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=3)  # AlexNet
clf_2 = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=3)  # VGG16
clf_3 = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=3)  # VGG19
print(f'Three classifiers created:\n{clf_1}')
print('-'*25)
"""
# Classifier (Adaboost) instantiation with GridSearch
param_grid = {
    'n_estimators': [50, 100, 200, 500],
    'learning_rate': [5, 1, 0.1, 0.01, 0.001, 0.0001],
    'algorithm': ['SAMME', 'SAMME.R']
    }
clf_1 = GridSearchCV(AdaBoostClassifier(), param_grid, refit=True, verbose=3)  # AlexNet
clf_2 = GridSearchCV(AdaBoostClassifier(), param_grid, refit=True, verbose=3)  # VGG16
clf_3 = GridSearchCV(AdaBoostClassifier(), param_grid, refit=True, verbose=3)  # VGG19
print(f'Three classifiers created:\n{clf_1}')
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

# Training loops
print('\nDeep features extraction...', end=' ')
alexnet_features_list, vgg16_features_list, vgg19_features_list = [], [], []
for (image, label) in train_dataloader:
    image = torch.Tensor(image.numpy().astype(np.double)).to(device)

    alexnet_features = alexnet(image)
    alexnet_features_list.append(alexnet_features.squeeze().cpu().detach().numpy())

    vgg16_features = vgg16(image)
    vgg16_features_list.append(vgg16_features.squeeze().cpu().detach().numpy())

    vgg19_features = vgg16(image)
    vgg19_features_list.append(vgg19_features.squeeze().cpu().detach().numpy())

alexnet_features_array = np.array(alexnet_features_list)
vgg16_features_array = np.array(vgg16_features_list)
vgg19_features_array = np.array(vgg19_features_list)
print('done')

print('\nTraining classifiers...')

clf_1.fit(alexnet_features_array, train_labels)
print('> clf_1 trained on AlexNet features')
print(f'Best parameters: {clf_1.best_params_}\n')

clf_2.fit(vgg16_features_array, train_labels)
print('> clf_2 trained on VGG16 features')
print(f'Best parameters: {clf_2.best_params_}\n')

clf_3.fit(vgg19_features_array, train_labels)
print('> clf_3 trained on VGG19 features')
print(f'Best parameters: {clf_3.best_params_}')
print('-'*25)


# TESTING
print('\n> TESTING PHASE')

# Load nodules ROIs and labels
test_rois, test_labels = load_nodules(TEST_DIR)
print('Loaded test nodules')
print(f'ROIs: {test_rois.shape}, labels: {test_labels.shape}')

# Create dataset and dataloader
test_dataset = SlicesDataset(test_rois, test_labels, transform=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Testing loops
alexnet_features_list, vgg16_features_list, vgg19_features_list = [], [], []
y_true, alexnet_predictions, vgg16_predictions, vgg19_predictions = [], [], [], []

print('Prediction loop...', end=' ')
for i, (image, label) in enumerate(test_dataloader):
    image = torch.Tensor(image.numpy().astype(np.double)).to(device)

    alexnet_features = alexnet(image)
    alexnet_features_list.append(alexnet_features.squeeze().cpu().detach().numpy())

    vgg16_features = vgg16(image)
    vgg16_features_list.append(vgg16_features.squeeze().cpu().detach().numpy())

    vgg19_features = vgg16(image)
    vgg19_features_list.append(vgg19_features.squeeze().cpu().detach().numpy())

    # Compute nodule prediction with MAX VOTING of slices prediction
    if (i + 1) % N_SLICES == 0:
        alexnet_features_array = np.array(alexnet_features_list)
        xSlice_predictions = clf_1.predict(alexnet_features_array)
        alexnet_predictions.append(np.bincount(xSlice_predictions).argmax())

        vgg16_features_array = np.array(vgg16_features_list)
        xSlice_predictions = clf_2.predict(vgg16_features_array)
        vgg16_predictions.append(np.bincount(xSlice_predictions).argmax())

        vgg19_features_array = np.array(vgg19_features_list)
        xSlice_predictions = clf_3.predict(vgg19_features_array)
        vgg19_predictions.append(np.bincount(xSlice_predictions).argmax())
        
        y_true.append(test_labels[i])

        alexnet_features_list, vgg16_features_list, vgg19_features_list = [], [], []

print('done')

# Compute final nodule prediction
y_pred = []
for preds in zip(alexnet_predictions, vgg16_predictions, vgg19_predictions):
    final_pred = sum(preds) / len(preds)
    final_pred = 1 if final_pred >= 0.5 else 0
    y_pred.append(final_pred)

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
print(f'> Number of correct nodules: {(np.array(y_true) == np.array(y_pred)).sum()}/200')

print('\nEND')
