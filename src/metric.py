
import numpy as np
from sklearn.metrics import f1_score
from . import load_data


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)


# One hot encoding, see here:
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
def one_hot_encoding(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

# def ydict2array(ydict):
#    yarray = np.array([],dtype=np.int8)
#    for station in ydict :
#        yarray = np.append(yarray, ydict[station], axis = 0)
#    return np.array(yarray)


def mean_average_presicion_score(y_true, y_scores, n_classes: int) -> float:
    #    y_true_array = ydict2array(y_true)
    #    y_scores_array = ydict2array(y_scores)
    #    return average_precision_score(one_hot_encoding(y_true_array, n_classes), one_hot_encoding(y_scores_array,n_classes))
    #    return f1_score(one_hot_encoding(y_true, n_classes), one_hot_encoding(y_scores,n_classes),average='weighted')
    if y_scores.ndim == 1:
        return f1_score(y_true, y_scores, average='weighted')
    else:
        return f1_score(one_hot_encoding(y_true, n_classes), one_hot_encoding(y_scores, n_classes), average='weighted')


def apply_my_metric(zipfile, y_true, n_classes=10):
    pred = load_data.load_yfile(zipfile)
    score = mean_average_presicion_score(y_true, pred, n_classes)
    return score
