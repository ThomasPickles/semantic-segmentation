# -*- coding: utf-8 -*-
# =============================================================================
# Description :
# Exemple d'utilisation de la Datachallenge
# =============================================================================
from sklearn.linear_model import LinearRegression
import numpy as np
import os

import datachallengecode.sources.load_data as load_data
import datachallengecode.sources.metric as metric


class2idx = {'beams': 1, 'cabletrays': 2, 'civils': 3, 'gratings': 4, 'guardrails': 5, 'hvac': 6, 'ladders': 7,
             'piping': 8, 'supports': 9, 'valves': 0, 'electric-boxes': 0, 'miscellaneous': 0, 'electric-equipments': 0}

# load the sample file with the indice 1 and its ground truth file
# PATH_x = r'./train/SCAN_1.ply'
# PATH_x = r'data/xtrain/'
PATH_x = r'./datachallengecode/data/xtrain.zip'
PATH_y = r'./datachallengecode/data/ytrain.csv'

# x_train
xtrain = load_data.load_xfile(PATH_x)
# ground truth y_train of the sample x_train
ytrain = load_data.load_yfile(PATH_y)
print('Cloud size : ', load_data.get_nbpoints(xtrain), ' points')

station2index = load_data.mapping_station2index('./ytrain_map_ind_station.csv')

# concatenate point cloud with ground truth for visualisation purpose
# with Cloud Compare software for instance
Generate_visu = False
if Generate_visu:
    savedir = r'./datachallengecode/data/visu'
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    load_data.save_ptcloud_withGT(xtrain, ytrain, station2index, savedir)

# =============================================================================
#     Test
# =============================================================================
# results simulation
n_classes = 10
y_scores = np.empty((0, n_classes))
indices = np.empty((0))
n_stations = len(list(station2index.keys()))

# xtrain is dict of 0: arr and 1:arr
for station in xtrain:
    ind_s, ind_e = station2index[station]
    n_points = ytrain[ind_s:ind_e+1].shape[0]
    y_outputs = np.random.normal(size=(n_points, n_classes))
    y_scores = np.concatenate([y_scores, metric.softmax(y_outputs)])
    indices = np.concatenate([indices, np.arange(ind_s, ind_e+1)])

savepath = r'./datachallengecode/data/y_random.csv'
load_data.save_predictions(y_scores, indices, savepath, onehot=True)

score = metric.apply_my_metric(savepath, ytrain)
print(score)

# =============================================================================
# Benchmark
# =============================================================================
n_features = 4

x = np.empty((0, n_features))
for station in xtrain:
    x = np.concatenate([x, xtrain[station][:, 3:]])
print(f"length of x is {len(x)}, length of y is {len(ytrain)}")
reg = LinearRegression().fit(x, ytrain)

PATH_xtest = r'./datachallengecode/data/xtest.zip'

xtest = load_data.load_xfile(PATH_xtest)

x = np.empty((0, n_features))
for station in xtest:
    x = np.concatenate([x, xtest[station][:, 3:]])

PATH_ytest = r'./datachallengecode/data/ytest.csv'
ytest = load_data.load_yfile(PATH_ytest)

path, fname = os.path.split(PATH_xtest)

station2index = load_data.mapping_station2index(
    os.path.join(path, 'ytest_map_ind_station.csv'))
y_pred = reg.predict(x)
y_indices = load_data.mappingindex2_list(station2index)

savepath = r'./datachallengecode/data/y_pred.csv'
load_data.save_predictions(y_pred, y_indices, savepath, onehot=False)

score = metric.apply_my_metric(savepath, ytest)
print(score)
