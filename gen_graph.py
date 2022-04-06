import numpy as np
import matplotlib.pyplot as plt
import os

#hyperparameters
DIR_FIG = "figures/"
X_LABEL = "Regularization parameter"
resolution_fig = 500

def file_to_array(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        f.close()

    n = len(lines)
    X = np.empty((n))
    for i,el in enumerate(lines):
        X[i] = float(el)
    return X

if __name__ == '__main__':

    plt.xlabel(X_LABEL)
    plt.ylabel("Score obtained")

    plt.xscale('log')
    X = file_to_array("measured")
    plt.plot(X, file_to_array("test_error"), label='test score')
    plt.plot(X, file_to_array("train_error"), label='training score')
    plt.legend()

    if not os.path.exists(DIR_FIG):
        os.makedirs(DIR_FIG)
    fig_name = f"{X_LABEL}({len(X)}).png"
    plt.savefig(DIR_FIG+fig_name, dpi=resolution_fig)