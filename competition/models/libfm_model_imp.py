# coding=utf-8
__author__ = 'songquanwang'

import numpy as np
import pandas as pd

from competition.inter.model_inter import ModelInter
import csv
import os
import cPickle
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import hstack
## sklearn
from sklearn.base import BaseEstimator
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.linear_model import Ridge, Lasso, LassoLars, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
# 梯度自举树，也是gdbt的实现
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
## hyperopt
from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
## keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils
## sklearn
# 梯度自举树，也是gdbt的实现
## hyperopt
## keras
import competition.conf.model_params_conf as model_param_conf
import competition.utils.utils as utils

from competition.conf.param_config import config

global trial_counter
global log_handler


class GbdtModelImp(ModelInter):
    def __init__(self, param, feat_folder, feat_name):
        super(ModelInter, self).__init__(param, feat_folder, feat_name)
        # 初始化run fold的各种集合的矩阵
        self.run_fold_matrix = np.empty((config.n_runs, config.n_folds), dtype=object)

    def train_predict(self, matrix, all=False):
        """
        数据训练
        :param train_end_date:
        :return:
        """
        ## scale
        scaler = StandardScaler()
        X_train = matrix.X_train.toarray()
        X_train[matrix.index_base] = scaler.fit_transform(X_train[matrix.index_base])
        ## dump feat
        dump_svmlight_file(X_train[matrix.index_base], matrix.labels_train[matrix.index_base], self.feat_train_path + ".tmp")
        if all:
            X_test = matrix.X_test.toarray()
            X_test = scaler.transform(X_test)
            dump_svmlight_file(X_test, matrix.labels_test, matrix.feat_test_path + ".tmp")

            ## train fm
            cmd = "%s -task r -train %s -test %s -out %s -dim '1,1,%d' -iter %d > libfm.log" % ( \
                model_param_conf.libfm_exe, matrix.feat_train_path + ".tmp", matrix.feat_test_path + ".tmp", matrix.raw_pred_test_path, \
                matrix.param['dim'], matrix.param['iter'])
            os.system(cmd)
            os.remove(matrix.feat_train_path + ".tmp")
            os.remove(matrix.feat_test_path + ".tmp")
            ## extract libfm prediction
            pred = np.loadtxt(matrix.raw_pred_test_path, dtype=float)
            ## labels are in [0,1,2,3]
            pred += 1
        else:
            X_valid = matrix.X_valid.toarray()
            X_valid = scaler.transform(X_valid)
            dump_svmlight_file(X_valid, matrix.labels_valid, matrix.feat_valid_path + ".tmp")

            ## train fm
            cmd = "%s -task r -train %s -test %s -out %s -dim '1,1,%d' -iter %d > libfm.log" % ( \
                model_param_conf.libfm_exe, matrix.feat_train_path + ".tmp", matrix.feat_valid_path + ".tmp", matrix.raw_pred_valid_path, \
                matrix.param['dim'], matrix.param['iter'])
            os.system(cmd)
            os.remove(matrix.feat_train_path + ".tmp")
            os.remove(matrix.feat_valid_path + ".tmp")
            ## extract libfm prediction
            pred = np.loadtxt(matrix.raw_pred_valid_path, dtype=float)
            ## labels are in [0,1,2,3]
            pred += 1

        return pred

    def get_predicts(self):
        return

    @staticmethod
    def get_id():
        return "gdbt_model_id"

    @staticmethod
    def get_name():
        return "gdbt_model"
