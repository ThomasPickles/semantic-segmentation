# -*- coding: utf-8 -*-
# =============================================================================
# Description :
# Exemple d'utilisation de la Datachallenge
# =============================================================================
import argparse
from sklearn.linear_model import LinearRegression
import numpy as np
import os
from src import load_data
# from src import metric
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

class2idx = {'beams': 1, 'cabletrays': 2, 'civils': 3, 'gratings': 4, 'guardrails': 5, 'hvac': 6, 'ladders': 7,
             'piping': 8, 'supports': 9, 'valves': 0, 'electric-boxes': 0, 'miscellaneous': 0, 'electric-equipments': 0}


def get_combined_station_data(x_by_station, target, n_features, points_by_station):

    y = np.empty((0))
    X = np.empty((0, n_features))

    for station in x_by_station:
        ind_s, ind_e = points_by_station[station]
        station_X = x_by_station[station]
        X = np.concatenate([X, station_X])
        y = np.concatenate([y, target[ind_s:ind_e+1]])
    print(f"dim of X is {X.shape}")
    print(f"dim of y is {y.shape}")
    return X, y


def get_trained_model(x_by_station, target, points_by_station, n_features):

    print('Training on : ', load_data.get_nbpoints(x_by_station), ' points')

    X, y = get_combined_station_data(
        x_by_station, target, n_features, points_by_station)

    # Always scale the input. The most convenient way is to use a pipeline.
    clf = make_pipeline(StandardScaler(),
                        SGDClassifier(max_iter=1000, tol=1e-3))
    clf.fit(X, y)

    print("Finished training.\n")
    return clf


def test_model(clf, visualise, n_features, target, points_by_station, x_by_station):

    print('Testing on : ', load_data.get_nbpoints(x_by_station), ' points')

    X, y = get_combined_station_data(
        x_by_station, target, n_features, points_by_station)

    predictions = clf.predict(X)
    print("Finished making predictions.\n")

    path_predictions = "predictions.csv"
    with open(path_predictions, "w") as f:
        print(f"Writing predictions and truth values to {path_predictions}")
        for hi, yi in zip(predictions, y):
            f.write(f"{hi},{yi}\n")

    # concatenate point cloud with ground truth for visualisation purpose
    # with Cloud Compare software for instance
    if visualise:
        print("Generating Cloud Compare files")
        load_data.save_ptcloud_withGT(
            x_by_station, y, points_by_station, 'visu/true')
        load_data.save_ptcloud_withGT(
            x_by_station, predictions, points_by_station, 'visu/predictions')

    return()


def train_and_test(dir, visualise):

    PATH_y = f"{dir}/y.csv"
    path_datamap = f"mapper.csv"
    path_x_train = f"{dir}/train"
    path_x_test = f"{dir}/test"

    target_values = load_data.load_yfile(PATH_y)  # ground truth
    points_by_station = load_data.mapping_station2index(path_datamap)
    xtrain_by_station = load_data.load_xfile(path_x_train)
    xtest_by_station = load_data.load_xfile(path_x_test)

    n_features = 7

    clf = get_trained_model(
        xtrain_by_station, target_values, points_by_station, n_features)

    test_model(clf, visualise, n_features, target_values,
               points_by_station, xtest_by_station)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train model.')
    parser.add_argument('-p', '--data-path', dest="data", type=str,
                        default="./data-sample",
                        help='Specify the path to the data folder')
    parser.add_argument('--no-vis', dest="vis", action='store_false',
                        help='Dont generate CloudCompare files')
    args = parser.parse_args()

    train_and_test(args.data, args.vis)
