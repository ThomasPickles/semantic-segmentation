# -*- coding: utf-8 -*-
# =============================================================================
# Description :
# Exemple d'utilisation de la Datachallenge
# =============================================================================
import argparse
from sklearn.linear_model import LinearRegression
import numpy as np
from src import load_data
from src import metric
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

class2idx = {'beams': 1, 'cabletrays': 2, 'civils': 3, 'gratings': 4, 'guardrails': 5, 'hvac': 6, 'ladders': 7,
             'piping': 8, 'supports': 9, 'valves': 0, 'electric-boxes': 0, 'miscellaneous': 0, 'electric-equipments': 0}


def get_combined_station_data(x_by_station, target, points_map):

    # get first pointset and just query its shape
    first_pointset = next(iter(x_by_station.values()))
    n_features = first_pointset.shape[1]

    y = np.empty((0))
    X = np.empty((0, n_features))

    for station in x_by_station:
        ind_s, ind_e = points_map[station]
        station_X = x_by_station[station]
        X = np.concatenate([X, station_X])
        y = np.concatenate([y, target[ind_s:ind_e+1]])
    print(f"dim of X is {X.shape}")
    print(f"dim of y is {y.shape}")
    return X, y


def get_trained_model(x_by_station, target, points_map):

    print('Training on : ', load_data.get_nbpoints(x_by_station), ' points')

    X, y = get_combined_station_data(
        x_by_station, target, points_map)

    # Always scale the input. The most convenient way is to use a pipeline.
    model = make_pipeline(StandardScaler(),
                          SGDClassifier(max_iter=1000, tol=1e-3))
    model.fit(X, y)

    print("Finished training.\n")
    return model


def test_model(x_by_station, target, points_map, model, visualise):

    print('Testing on : ', load_data.get_nbpoints(x_by_station), ' points')

    X, y = get_combined_station_data(
        x_by_station, target, points_map)

    predictions = model.predict(X)
    print("Finished making predictions.\n")

    score = metric.mean_average_presicion_score(y, predictions, 10)
    print(f"Score obtained : {score}")

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
            x_by_station, y, points_map, 'visu/true')
        load_data.save_ptcloud_withGT(
            x_by_station, predictions, points_map, 'visu/predictions')

    return()


def train_and_test(dir, visualise):

    path_mapper = f"mapper.csv"
    path_y = f"{dir}/y.csv"
    path_x_train = f"{dir}/train"
    path_x_test = f"{dir}/test"

    target_values = load_data.load_yfile(path_y)  # ground truth
    points_map = load_data.mapping_station2index(path_mapper)
    xtrain = load_data.load_xfile(path_x_train)
    xtest = load_data.load_xfile(path_x_test)

    model = get_trained_model(xtrain, target_values, points_map)

    test_model(xtest, target_values, points_map, model, visualise)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train model.')
    parser.add_argument('--full', dest="data", action='store_const',
                        const="./data-full", default="./data-sample",
                        help='Use all data (default is a sample)')
    parser.add_argument('--no-vis', dest="vis", action='store_false',
                        help='Dont generate CloudCompare files')
    args = parser.parse_args()

    train_and_test(args.data, args.vis)
