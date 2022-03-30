# -*- coding: utf-8 -*-

# https://github.com/dranjan/python-plyfile
from typing import Any, Dict
import plyfile as ply
import numpy as np
import os
import shutil

class2idx = {'beams': 1, 'cabletrays': 2, 'civils': 3, 'gratings': 4, 'guardrails': 5, 'hvac': 6, 'ladders': 7,
             'piping': 8, 'supports': 9, 'valves': 0, 'electric-boxes': 0, 'miscellaneous': 0, 'electric-equipments': 0}

# =============================================================================
# load and save data
# =============================================================================


def read_x_plyfile(plypath):
    plydata = ply.PlyData.read(plypath)
    coord = np.array([plydata['points']['x'], plydata['points']
                     ['y'], plydata['points']['z']]).T
    rgb = np.array([plydata['rgb']['r'], plydata['rgb']
                   ['g'], plydata['rgb']['b']]).T
    intensity = np.reshape(np.array(plydata['intensity']['i']), (-1, 1))
    fname = os.path.splitext(plypath)[0]
    station_id = np.uint8(fname.split('_')[1])
    return station_id, np.hstack((coord, rgb, intensity))

# load sample files


def load_xfile(path: str) -> Dict[int, Any]:
    x = {}
    if os.path.isdir(path):
        xfile_list = [file for file in os.listdir(path) if file[-3:] == 'ply']
        for xfile in xfile_list:
            station_id, features = read_x_plyfile(os.path.join(path, xfile))
            x[station_id] = features
    elif os.path.isfile(path):
        ext = os.path.splitext(path)[1]
        if ext == '.ply':
            station_id, features = read_x_plyfile(path)
            x[station_id] = features
        elif ext == '.zip':
            extract_dir = os.path.join(os.path.split(
                path)[0], os.path.split(path)[1][:-4])
            if not os.path.exists(extract_dir):
                os.mkdir(extract_dir)
                shutil.unpack_archive(path, extract_dir)
            xfile_list = os.listdir(extract_dir)
            for xfile in xfile_list:
                station_id, features = read_x_plyfile(
                    os.path.join(extract_dir, xfile))
                x[station_id] = features
    return x


def load_yfile(path):
    y = []
    with open(path, 'r') as csvfile:
        next(csvfile)  # skip header
        y = [int(row.split(',')[1]) for row in csvfile]
    return np.array(y)


def mapping_station2index(path):
    if not os.path.exists(path):
        raise FileNotFoundError
    station2index = {}
    with open(path, 'r') as csvfile:
        next(csvfile)  # skip header
        for row in csvfile:
            data = [int(elt) for elt in row.split(',')]
            station2index[data[0]] = np.array([data[1], data[2]])
    return station2index


def save_predictions(ypred, indices, savepath, onehot=False):
    print('Warning : the prediction values are converted into integer')
    with open(savepath, 'w') as f:
        f.write('ID,class\n')
        for i, p in enumerate(ypred):
            if onehot:
                predscal = np.argmax(p)  # onehot into scalar prediction
            else:
                predscal = int(np.round(p))
            f.write(str(int(indices[i])) + ','+str(predscal))
            f.write('\n')


def mappingindex2_list(station2index):
    m2i_list = np.empty((0))
    for station in station2index:
        ind_s, ind_e = station2index[station]
        m2i_list = np.concatenate([m2i_list, np.arange(ind_s, ind_e+1)])
    return m2i_list

# Concatenate point cloud with its ground truth and save it


def save_ptcloud_withGT(x: Dict, y: np.ndarray, station2index, savedir):
    # load the point cloud coordinate and its features

    if not os.path.exists(savedir):
            os.makedirs(savedir)

    for station in x:
        features = x[station]
        classes = y
        coord = features[:, :3]
        rgb = features[:, 3:6]
        intensity = features[:, 6]

        coord = np.array([tuple(x) for x in coord], dtype=[
                         ('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        rgb = np.array([tuple(x) for x in rgb], dtype=[
                       ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
        intensity = np.array(intensity, dtype=[('i', 'u1')])
        classes = np.array(classes, dtype=[('c', 'u1')])

        # Build PLY structures
        coord = ply.PlyElement.describe(coord, 'points')
        rgb = ply.PlyElement.describe(rgb, 'rgb')
        intensity = ply.PlyElement.describe(intensity, 'intensity')
        classes = ply.PlyElement.describe(classes, 'class')

        # save the file in PLY format
        outpath = os.path.join(savedir, 'SCAN_'+str(station)+'_VISU.ply')
        ply.PlyData([coord, rgb, intensity, classes]).write(outpath)


def get_nbpoints(x):
    n_points = 0
    for station in x:
        n_points += x[station].shape[0]
    return n_points
