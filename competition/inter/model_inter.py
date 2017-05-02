# -*- coding: utf-8 -*-
__author__ = 'songquanwang'

import abc
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
import competition.conf.model_params_conf as model_param_conf
import competition.utils.utils as utils
from scipy.sparse import hstack


class ModelInter(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, param, feat_folder, feat_name):
        self.param = param
        self.feat_folder = feat_folder
        self.feat_name = feat_name

    @abc.abstractmethod
    def get_predicts(self, start_date, period, period_number, step=1, in_one_dt=False):
        """

        :param start_date:
        :param period:
        :param period_number:
        :param step:
        :param in_one_dt:
        :return:
        """
        return

    def create_train_valid(self, path, all=False):

        # feat
        feat_train_path = "%s/train.feat" % path
        feat_valid_path = "%s/valid.feat" % path
        if all:
            feat_valid_path = "%s/test.feat" % path
        X_train, labels_train = load_svmlight_file(feat_train_path)
        X_valid, labels_valid = load_svmlight_file(feat_valid_path)
        # 延展array
        if X_valid.shape[1] < X_train.shape[1]:
            X_valid = hstack([X_valid, np.zeros((X_valid.shape[0], X_train.shape[1] - X_valid.shape[1]))])
        elif X_valid.shape[1] > X_train.shape[1]:
            X_train = hstack([X_train, np.zeros((X_train.shape[0], X_valid.shape[1] - X_train.shape[1]))])
        X_train = X_train.tocsr()
        X_valid = X_valid.tocsr()

        return (X_train, labels_train), (X_valid, labels_valid)

    def create_weight(self, path, all=False):
        # weight load weight -train
        weight_train_path = "%s/train.feat.weight" % path
        weight_train = np.loadtxt(weight_train_path, dtype=float)
        result = weight_train
        # weight load weight -valid
        if all == False:
            weight_valid_path = "%s/valid.feat.weight" % path
            weight_valid = np.loadtxt(weight_valid_path, dtype=float)
            result = (weight_train, weight_valid)

        return result

    def create_info(self, path, all=False):

        # info
        info_train_path = "%s/train.info" % path
        info_valid_path = "%s/valid.info" % path
        if all:
            info_valid_path = "%s/test.info" % path
        # load info
        info_train = pd.read_csv(info_train_path)
        info_valid = pd.read_csv(info_valid_path)

        return info_train, info_valid

    def create_cdf(self, path, all=False):
        # cdf
        cdf_valid_path = "%s/valid.cdf" % path
        if all:
            cdf_valid_path = "%s/test.cdf" % path
        ## load cdf
        cdf_valid = np.loadtxt(cdf_valid_path, dtype=float)

        return cdf_valid

    def create_watch_list(self, dtrain_base, dvalid_base, all=False):
        watchlist = []
        if model_param_conf.verbose_level >= 2:
            if all:
                watchlist = [(dtrain_base, 'train')]
            else:
                watchlist = [(dtrain_base, 'train'), (dvalid_base, 'valid')]
        return watchlist

    def create_base_all(self, X_train, labels_train, weight_train, X_test, labels_test, numTrain, weight_valid):

        # 分割训练数据
        index_base, index_meta = utils.bootstrap_all(model_param_conf.bootstrap_replacement, numTrain, model_param_conf.bootstrap_ratio)
        dtrain = xgb.DMatrix(X_train[index_base], label=labels_train[index_base], weight=weight_train[index_base])
        dtest = xgb.DMatrix(X_test, label=labels_test)

        return dtrain, dtest

    def create_base_run_fold(self, run, fold, X_train, labels_train, weight_train, X_valid, labels_valid, numTrain, weight_valid):

        # 分割训练数据
        index_base, index_meta = utils.create_base_run_fold(model_param_conf.bootstrap_replacement, run, fold, numTrain, model_param_conf.bootstrap_ratio)
        dtrain_base = xgb.DMatrix(X_train[index_base], label=labels_train[index_base], weight=weight_train[index_base])
        dvalid_base = xgb.DMatrix(X_valid, label=labels_valid, weight=weight_valid)

        return dtrain_base, dvalid_base,

    @abc.abstractmethod
    def get_id(self):
        return

    @abc.abstractmethod
    def get_name(self):
        return

    def pre_process(self):
        return

    def optmize(self):
        return

    def train(self):
        return

    def predict(self):
        return
