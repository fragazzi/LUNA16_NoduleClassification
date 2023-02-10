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
    'kernel': ['poly'] #['linear', 'rbf']
    }
clf_1 = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=3)  # AlexNet
clf_2 = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=3)  # VGG16
clf_3 = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=3)  # VGG19
print(f'Three classifiers created:\n{clf_1}')
print('-'*25)

"""
# Classifier (Adaboost) instantiation with GridSearch
param_grid = {
    'n_estimators': [100, 200, 500],
    'learning_rate': [1, 0.1, 0.01, 0.001, 0.0001],
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
# AlexNet
print('\nTraining classifier with AlexNet features extractor')
train_features_list = []
for (image, label) in train_dataloader:
    image = torch.Tensor(image.numpy().astype(np.double)).to(device)

    features = alexnet(image)
    train_features_list.append(features.squeeze().cpu().detach().numpy())

train_features_array = np.array(train_features_list)

print('Features extraction loop finished')
print(f'Training features array shape: {train_features_array.shape}')

clf_1.fit(train_features_array, train_labels)
print('> clf_1 trained on AlexNet features')
print(f'Best parameters: {clf_1.best_params_}')

# VGG16
print('\nTraining classifier with VGG16 features extractor')
train_features_list = []
for (image, label) in train_dataloader:
    image = torch.Tensor(image.numpy().astype(np.double)).to(device)

    features = vgg16(image)
    train_features_list.append(features.squeeze().cpu().detach().numpy())

train_features_array = np.array(train_features_list)

print('Features extraction loop finished')
print(f'Training features array shape: {train_features_array.shape}')

clf_2.fit(train_features_array, train_labels)
print('> clf_2 trained on VGG16 features')
print(f'Best parameters: {clf_2.best_params_}')

# VGG19
print('\nTraining classifier with VGG19 features extractor')
train_features_list = []
for (image, label) in train_dataloader:
    image = torch.Tensor(image.numpy().astype(np.double)).to(device)

    features = vgg19(image)
    train_features_list.append(features.squeeze().cpu().detach().numpy())

train_features_array = np.array(train_features_list)

print('Features extraction loop finished')
print(f'Training features array shape: {train_features_array.shape}')

clf_3.fit(train_features_array, train_labels)
print('> clf_3 trained on VGG19 features')
print(f'Best parameters: {clf_3.best_params_}')
print('-'*25)


# TESTING PHASE
print('\n> TESTING PHASE')

# Load nodules ROIs and labels
test_rois, test_labels = load_nodules(TEST_DIR)
print('Loaded train nodules')
print(f'ROIs: {test_rois.shape}, labels: {test_labels.shape}')

# Create dataset and dataloader
test_dataset = SlicesDataset(test_rois, test_labels, transform=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# Testing loops
y_true, alexnet_predictions, vgg16_predictions, vgg19_predictions = [], [], [], []

# AlexNet
print('\nPrediction with AlexNet features extractor...', end=' ')
test_features_list = []
for i, (image, label) in enumerate(test_dataloader):
    image = torch.Tensor(image.numpy().astype(np.double)).to(device)

    features = alexnet(image)
    test_features_list.append(features.squeeze().cpu().detach().numpy())

    # Compute nodule prediction with MAX VOTING of slices prediction
    if (i + 1) % N_SLICES == 0:
        test_features_array = np.array(test_features_list)
        xSlice_predictions = clf_1.predict(test_features_array)

        nodule_prediction = np.bincount(xSlice_predictions).argmax()

        alexnet_predictions.append(nodule_prediction)
        y_true.append(test_labels[i])

        test_features_list = []
print('done')

# VGG16
print('\nPrediction with VGG16 features extractor...', end=' ')
test_features_list = []
for i, (image, label) in enumerate(test_dataloader):
    image = torch.Tensor(image.numpy().astype(np.double)).to(device)

    features = vgg16(image)
    test_features_list.append(features.squeeze().cpu().detach().numpy())

    # Compute nodule prediction with MAX VOTING of slices prediction
    if (i + 1) % N_SLICES == 0:
        test_features_array = np.array(test_features_list)
        xSlice_predictions = clf_2.predict(test_features_array)

        nodule_prediction = np.bincount(xSlice_predictions).argmax()
        vgg16_predictions.append(nodule_prediction)

        test_features_list = []
print('done')

# VGG19
print('\nPrediction with VGG19 features extractor...', end=' ')
test_features_list = []
for i, (image, label) in enumerate(test_dataloader):
    image = torch.Tensor(image.numpy().astype(np.double)).to(device)

    features = vgg19(image)
    test_features_list.append(features.squeeze().cpu().detach().numpy())

    # Compute nodule prediction with MAX VOTING of slices prediction
    if (i + 1) % N_SLICES == 0:
        test_features_array = np.array(test_features_list)
        xSlice_predictions = clf_3.predict(test_features_array)

        nodule_prediction = np.bincount(xSlice_predictions).argmax()
        vgg19_predictions.append(nodule_prediction)

        test_features_list = []
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
