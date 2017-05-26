# coding:utf-8
"""
__file__

    init.py

__description__
   init path


__author__

    songquanwang

"""

import os

import competition.conf.model_params_conf as config


def init_path():
    # create feat folder
    if not os.path.exists(config.feat_folder):
        os.makedirs(config.feat_folder)

    # creat folder for the training and testing feat
    if not os.path.exists("%s/All" % config.feat_folder):
        os.makedirs("%s/All" % config.feat_folder)

    # creat folder for each run and fold
    for run in range(1, config.n_runs + 1):
        for fold in range(1, config.n_folds + 1):
            path = "%s/Run%d/Fold%d" % (config.feat_folder, run, fold)
            if not os.path.exists(path):
                os.makedirs(path)
