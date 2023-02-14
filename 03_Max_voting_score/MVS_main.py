# FUSION BASED ON MAXIMUM VOTING SCORE
# Third technique proposed in the paper

import numpy as np

from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score

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

# Classifiers (SVM and ADABOOST) instantiations with GridSearch
# SVM
svm_param_grid = {
    'C': [0.01, 0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['linear', 'rbf']
    }
svm_1 = GridSearchCV(svm.SVC(), svm_param_grid, refit=True, verbose=3)
svm_2 = GridSearchCV(svm.SVC(), svm_param_grid, refit=True, verbose=3)
svm_3 = GridSearchCV(svm.SVC(), svm_param_grid, refit=True, verbose=3)
svm_4 = GridSearchCV(svm.SVC(), svm_param_grid, refit=True, verbose=3)
svm_5 = GridSearchCV(svm.SVC(), svm_param_grid, refit=True, verbose=3)
print(f'Five SVM classifier created:\n{svm_1}')
# ADABOOST
adaboost_param_grid = {
    'n_estimators': [50, 100, 200, 500],
    'learning_rate': [5, 1, 0.1, 0.01, 0.001, 0.0001],
    'algorithm': ['SAMME', 'SAMME.R']
    }
adaboost_1 = GridSearchCV(AdaBoostClassifier(), svm_param_grid, refit=True, verbose=3)
adaboost_2 = GridSearchCV(AdaBoostClassifier(), svm_param_grid, refit=True, verbose=3)
adaboost_3 = GridSearchCV(AdaBoostClassifier(), svm_param_grid, refit=True, verbose=3)
adaboost_4 = GridSearchCV(AdaBoostClassifier(), svm_param_grid, refit=True, verbose=3)
adaboost_5 = GridSearchCV(AdaBoostClassifier(), svm_param_grid, refit=True, verbose=3)
print(f'Five ADABOOST classifier created:\n{adaboost_1}')
print('-'*25)


# TRAINING
print('\n> TRAINING PHASE')

# Load nodules ROIs and labels
train_rois, train_labels = load_nodules(TRAIN_DIR)
print('Loaded train nodules')
print(f'ROIs: {train_rois.shape}, labels: {train_labels.shape}')

# Create dataset and dataloader
train_dataset = SlicesDataset(train_rois, train_labels, transform=True)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

# Training loop
print('\nTraining loop...')
alexnet_features_list, vgg16_features_list, vgg19_features_list = [], [], []
averaged_features_list, concatenated_features_list = [], []

for (image, label) in train_dataloader:
    image = torch.Tensor(image.numpy().astype(np.double)).to(device)

    alexnet_features = alexnet(image).squeeze().cpu().detach().numpy()
    vgg16_features = vgg16(image).squeeze().cpu().detach().numpy()
    vgg19_features = vgg19(image).squeeze().cpu().detach().numpy()

    sum_features = alexnet_features + vgg16_features + vgg19_features
    avg_features = sum_features / 3

    concatenated_features = np.concatenate([alexnet_features, vgg16_features, vgg19_features])

    alexnet_features_list.append(alexnet_features)
    vgg16_features_list.append(vgg16_features)
    vgg19_features_list.append(vgg19_features)
    averaged_features_list.append(avg_features)
    concatenated_features_list.append(concatenated_features)

alexnet_features_array = np.array(alexnet_features_list)  # -> fv1
vgg16_features_array = np.array(vgg16_features_list)  # -> fv2
vgg19_features_array = np.array(vgg19_features_list)  # -> fv3
averaged_features_array = np.array(averaged_features_list)  # -> fv4
concatenated_features_array = np.array(concatenated_features_list)  # -> fv5

print('Features extraction loop finished')
print(f'Alexnet features array shape: {alexnet_features_array.shape}')
print(f'VGG16 features array shape: {vgg16_features_array.shape}')
print(f'VGG19 features array shape: {vgg19_features_array.shape}')
print(f'Averaged features array shape: {averaged_features_array.shape}')
print(f'Concatenated features array shape: {concatenated_features_array.shape}')

print('\nTraining classifiers...')
svm_1.fit(alexnet_features_array, train_labels)
adaboost_1.fit(alexnet_features_array, train_labels)
print('> svm_1 and adaboost_1 trained on AlexNet features')

svm_2.fit(vgg16_features_array, train_labels)
adaboost_2.fit(vgg16_features_array, train_labels)
print('> svm_2 and adaboost_2 trained on VGG16 features')

svm_3.fit(vgg19_features_array, train_labels)
adaboost_3.fit(vgg19_features_array, train_labels)
print('> svm_3 and adaboost_3 trained on VGG19 features')

svm_4.fit(averaged_features_array, train_labels)
adaboost_4.fit(averaged_features_array, train_labels)
print('> svm_4 and adaboost_4 trained on averaged features')

svm_5.fit(concatenated_features_array, train_labels)
adaboost_5.fit(concatenated_features_array, train_labels)
print('> svm_5 and adaboost_5 trained on concatenated features')


# TESTING PHASE
print('\n> TESTING PHASE')

# Load nodules ROIs and labels
test_rois, test_labels = load_nodules(TEST_DIR)
print('Loaded test nodules')
print(f'ROIs: {test_rois.shape}, labels: {test_labels.shape}')

# Create dataset and dataloader
test_dataset = SlicesDataset(test_rois, test_labels, transform=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Test classification loop
print('\nStarting test nodules classification loop...', end=' ')

alexnet_features_list, vgg16_features_list, vgg19_features_list = [], [], []
averaged_features_list, concatenated_features_list = [], []
y_true, y_pred = [], []

for i, (image, label) in enumerate(test_dataloader):
    image = torch.Tensor(image.numpy().astype(np.double)).to(device)

    alexnet_features = alexnet(image).squeeze().cpu().detach().numpy()  # -> fv1
    vgg16_features = vgg16(image).squeeze().cpu().detach().numpy()  # -> fv2
    vgg19_features = vgg19(image).squeeze().cpu().detach().numpy()  # -> fv3

    sum_features = alexnet_features + vgg16_features + vgg19_features
    avg_features = sum_features / 3  # -> fv4

    concatenated_features = np.concatenate([alexnet_features, vgg16_features, vgg19_features])  # -> fv5

    alexnet_features_list.append(alexnet_features)
    vgg16_features_list.append(vgg16_features)
    vgg19_features_list.append(vgg19_features)
    averaged_features_list.append(avg_features)
    concatenated_features_list.append(concatenated_features)

    # Compute nodule prediction with MAX VOTING of slices prediction
    if (i + 1) % N_SLICES == 0:

        alexnet_features_array = np.array(alexnet_features_list)
        pv1 = svm_1.predict(alexnet_features_array)
        pv1 = np.bincount(pv1).argmax()
        pv6 = adaboost_1.predict(alexnet_features_array)
        pv6 = np.bincount(pv6).argmax()

        vgg16_features_array = np.array(vgg16_features_list)
        pv2 = svm_2.predict(vgg16_features_array)
        pv2 = np.bincount(pv2).argmax()
        pv7 = adaboost_2.predict(vgg16_features_array)
        pv7 = np.bincount(pv7).argmax()

        vgg19_features_array = np.array(vgg19_features_list)
        pv3 = svm_1.predict(vgg19_features_array)
        pv3 = np.bincount(pv3).argmax()
        pv8 = adaboost_1.predict(vgg19_features_array)
        pv8 = np.bincount(pv8).argmax()

        averaged_features_array = np.array(averaged_features_list)
        pv4 = svm_1.predict(averaged_features_array)
        pv4 = np.bincount(pv4).argmax()
        pv9 = adaboost_1.predict(averaged_features_array)
        pv9 = np.bincount(pv9).argmax()

        concatenated_features_array = np.array(concatenated_features_list)
        pv5 = svm_1.predict(concatenated_features_array)
        pv5 = np.bincount(pv5).argmax()
        pv10 = adaboost_1.predict(concatenated_features_array)
        pv10 = np.bincount(pv10).argmax()

        nodule_prediction = pv1 + pv2 + pv3 + pv4 + pv5 + pv6 + pv7 + pv8 + pv9 + pv10
        nodule_prediction = 1 if nodule_prediction >= 5 else 0

        y_pred.append(nodule_prediction)
        y_true.append(test_labels[i])

        alexnet_features_list, vgg16_features_list, vgg19_features_list = [], [], []
        averaged_features_list, concatenated_features_list = [], []

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
print(f'> Correct nodules: {(np.array(y_true) == np.array(y_pred)).sum()}/200')

print('END')
