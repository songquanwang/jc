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

    def train_predict(self, set_obj,all =False):
        """
        数据训练
        :param train_end_date:
        :return:
        """
        X_train, X_test = set_obj.X_train.toarray(), set_obj.X_test.toarray()

        ## scale
        scaler = StandardScaler()
        X_train[set_obj.index_base] = scaler.fit_transform(X_train[set_obj.index_base])
        X_test = scaler.transform(X_test)

        ## dump feat
        dump_svmlight_file(X_train[set_obj.index_base], set_obj.labels_train[set_obj.index_base], self.feat_train_path + ".tmp")
        dump_svmlight_file(X_test, set_obj.labels_test, set_obj.feat_test_path + ".tmp")

        ## train fm
        cmd = "%s -task r -train %s -test %s -out %s -dim '1,1,%d' -iter %d > libfm.log" % ( \
            model_param_conf.libfm_exe, set_obj.feat_train_path + ".tmp", set_obj.feat_test_path + ".tmp", raw_pred_test_path, \
            param['dim'], param['iter'])
        os.system(cmd)
        os.remove(feat_train_path + ".tmp")
        os.remove(feat_test_path + ".tmp")

        ## extract libfm prediction
        pred = np.loadtxt(raw_pred_test_path, dtype=float)
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
