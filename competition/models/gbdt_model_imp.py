# coding=utf-8
__author__ = 'yangdongyue'

from competition.inter.model_inter import ModelInter

import sys
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

    def feature_all(self):
        path = "%s/All" % (self.feat_folder)
        (self.X_train, self.labels_train), (self.X_test, self.labels_test) = self.create_train_valid(path, all=True)
        self.weight_train = self.create_weight(path, all=True)
        self.info_train, self.info_test = self.create_info(path, all=True)
        self.numTrain = self.info_train.shape[0]
        self.numTest = self.info_test.shape[0]
        self.cdf_test = self.create_cdf(path, all=True)
        self.dtrain, self.dtest = self.create_base_all(self, self.X_train, self.labels_train, self.weight_train, self.X_valid, self.labels_valid, self.numTrain, self.weight_valid)
        self.create_watch_list = self.create_watch_list(self.dtrain, None, all=True)

    def feature_run_fold(self, run, fold):
        """
        每个run 每个fold 生成
        :param run:
        :param fold:
        :return:
        """
        path = "%s/Run%d/Fold%d" % (self.feat_folder, run, fold)
        (self.run_fold_matrix[run][fold].X_train, self.run_fold_matrix[run][fold].labels_train), (self.run_fold_matrix[run][fold].X_valid, self.run_fold_matrix[run][fold].labels_valid) = self.create_train_valid(path)
        self.run_fold_matrix[run][fold].weight_train, self.run_fold_matrix[run][fold].weight_valid = self.create_weight(path)
        self.run_fold_matrix[run][fold].info_train, self.run_fold_matrix[run][fold].info_valid = self.create_info(path)
        numTrain = self.run_fold_matrix[run][fold].info_train.shape[0]
        numValid = self.run_fold_matrix[run][fold].info_valid.shape[0]
        self.run_fold_matrix[run][fold].numTrain = numTrain
        self.run_fold_matrix[run][fold].numValid = numValid
        self.run_fold_matrix[run][fold].cdf_valid = self.create_cdf(path)
        self.run_fold_matrix[run][fold].dtrain_base, self.run_fold_matrix[run][fold].dvalid_base = self.create_base_run_fold(run, fold, self.X_train, self.labels_train, self.weight_train, self.X_valid, self.labels_valid,
                                                                                                                             numTrain, self.weight_valid)
        self.run_fold_matrix[run][fold].create_watch_list = self.create_watch_list(self.run_fold_matrix[run][fold].dtrain_base, self.run_fold_matrix[run][fold].dvalid_base)

    def optimize(self):
        """
        参数优化
        :param train_end_date:
        :return:
        """
        # 读取sku 与 一级品类对应关系 (item_sku_id,cate1)
        params = {}

        return params

    def train_predict(self, set_obj):
        """
        数据训练
        :param train_end_date:
        :return:
        """
        if self.param["task"] in ["regression", "ranking"]:
            pred = self.reg_rank_predict(set_obj)
        elif self.param["task"] in ["softmax"]:
            pred = self.soft_max_predict(set_obj)
        elif self.param["task"] in ["softkappa"]:
            pred = self.soft_softkappa_predict(set_obj)
        elif self.param["task"] in ["ebc"]:
            pred = self.ebc_predict(set_obj)
        elif self.param["task"] in ["cocr"]:
            pred = self.cocr_predict(set_obj)

        return pred

    def reg_rank_predict(self, set_obj):
        evalerror_regrank_valid = lambda preds, dtrain: utils.evalerror_regrank_cdf(preds, dtrain, set_obj.cdf_valid)
        bst = xgb.train(self.param, set_obj.dtrain_base, self.param['num_round'], set_obj.watchlist, feval=evalerror_regrank_valid)
        pred = bst.predict(set_obj.dvalid_base)
        return pred

    def soft_max_predict(self, set_obj):
        evalerror_softmax_valid = lambda preds, dtrain: utils.evalerror_softmax_cdf(preds, dtrain, set_obj.cdf_valid)
        ## softmax regression with xgboost
        bst = xgb.train(self.param, set_obj.dtrain_base, self.param['num_round'], set_obj.watchlist, feval=evalerror_softmax_valid)
        # (6688, 4)
        pred = bst.predict(set_obj.dvalid_base)
        w = np.asarray(range(1, model_param_conf.num_of_class + 1))
        # 加权相乘 ？累加
        pred = pred * w[np.newaxis, :]
        pred = np.sum(pred, axis=1)
        return pred

    def soft_softkappa_predict(self, set_obj):
        ## softkappa with xgboost
        obj = lambda preds, dtrain: utils.softkappaObj(preds, dtrain, hess_scale=self.param['hess_scale'])
        bst = xgb.train(self.param, set_obj.dtrain_base, self.param['num_round'], set_obj.watchlist, obj=obj, feval=utils.evalerror_softkappa_valid)
        pred = utils.softmax(bst.predict(set_obj.dvalid_base))
        w = np.asarray(range(1, model_param_conf.num_of_class + 1))
        pred = pred * w[np.newaxis, :]
        pred = np.sum(pred, axis=1)
        return pred

    def ebc_predict(self, set_obj):
        # ebc with xgboost
        obj = lambda preds, dtrain: utils.ebcObj(preds, dtrain)
        bst = xgb.train(self.param, set_obj.dtrain_base, self.param['num_round'], set_obj.watchlist, obj=obj, feval=utils.evalerror_ebc_valid)
        pred = utils.sigmoid(bst.predict(set_obj.dvalid_base))
        pred = utils.applyEBCRule(pred, hard_threshold=utils.ebc_hard_threshold)
        return pred

    def cocr_predict(self, set_obj):
        ## cocr with xgboost
        obj = lambda preds, dtrain: utils.cocrObj(preds, dtrain)
        bst = xgb.train(self.param, set_obj.dtrain_base, self.param['num_round'], set_obj.watchlist, obj=obj, feval=utils.evalerror_cocr_valid)
        pred = bst.predict(set_obj.dvalid_base)
        pred = utils.applyCOCRRule(pred)
        return pred

    def out_put_run_fold(self, run, fold, bagging, feat_name, trial_counter, kappa_valid, X_train, Y_valid, pred_raw, pred_rank, kappa_cv):
        save_path = "%s/Run%d/Fold%d" % (model_param_conf.output_path, run, fold)
        raw_pred_valid_path = "%s/valid.raw.pred.%s_[Id@%d].csv" % (save_path, feat_name, trial_counter)
        rank_pred_valid_path = "%s/valid.pred.%s_[Id@%d].csv" % (save_path, feat_name, trial_counter)
        kappa_cv[run - 1, fold - 1] = kappa_valid
        ## save this prediction
        dfPred = pd.DataFrame({"target": Y_valid, "prediction": pred_raw})
        dfPred.to_csv(raw_pred_valid_path, index=False, header=True,
                      columns=["target", "prediction"])
        ## save this prediction
        dfPred = pd.DataFrame({"target": Y_valid, "prediction": pred_rank})
        dfPred.to_csv(rank_pred_valid_path, index=False, header=True,
                      columns=["target", "prediction"])
        if (bagging + 1) != model_param_conf.bagging_size:
            print("              {:>3}   {:>3}   {:>3}   {:>6}   {} x {}".format(
                run, fold, bagging + 1, np.round(kappa_valid, 6), X_train.shape[0], X_train.shape[1]))
        else:
            print("                    {:>3}       {:>3}      {:>3}    {:>8}  {} x {}".format(
                run, fold, bagging + 1, np.round(kappa_valid, 6), X_train.shape[0], X_train.shape[1]))

    def out_put_all(self, feat_folder, feat_name, trial_counter, kappa_cv_mean, kappa_cv_std):
        path = "%s/All" % (feat_folder)
        save_path = "%s/All" % model_param_conf.output_path
        subm_path = "%s/Subm" % model_param_conf.output_path
        raw_pred_test_path = "%s/test.raw.pred.%s_[Id@%d].csv" % (save_path, feat_name, trial_counter)
        rank_pred_test_path = "%s/test.pred.%s_[Id@%d].csv" % (save_path, feat_name, trial_counter)
        # submission path (relevance as in [1,2,3,4])
        subm_path = "%s/Subm" % model_param_conf.output_path
        subm_path = "%s/test.pred.%s_[Id@%d]_[Mean%.6f]_[Std%.6f].csv" % (subm_path, feat_name, trial_counter, kappa_cv_mean, kappa_cv_std)
        ## write
        output = pd.DataFrame({"id": id_test, "prediction": pred_raw})
        output.to_csv(raw_pred_test_path, index=False)

        ## write
        output = pd.DataFrame({"id": id_test, "prediction": pred_rank})
        output.to_csv(rank_pred_test_path, index=False)

        ## write score
        pred_score = getScore(pred, cdf_test)
        output = pd.DataFrame({"id": id_test, "prediction": pred_score})
        output.to_csv(subm_path, index=False)

    def get_predicts(self):
        return

    @staticmethod
    def get_id():
        return "gdbt_model_id"

    @staticmethod
    def get_name():
        return "gdbt_model"

    def hyperopt_obj(self, feat_name, trial_counter):
        kappa_cv = np.zeros((config.n_runs, config.n_folds), dtype=float)
        for run in range(1, config.n_runs + 1):
            for fold in range(1, config.n_folds + 1):
                # 生成 run_fold_matrix
                self.feature_run_fold(self, run, fold)
                run_fold_matrix = self.run_fold_matrix[run][fold]
                preds_bagging = np.zeros((run_fold_matrix.numValid, model_param_conf.bagging_size), dtype=float)
                for n in range(model_param_conf.bagging_size):
                    pred = self.train_predict(run_fold_matrix)
                    ## weighted averageing over different models
                    pred_valid = pred
                    ## this bagging iteration
                    preds_bagging[:, n] = pred_valid
                    # 每次会把当前bagging的结果累计进来 求均值
                    pred_raw = np.mean(preds_bagging[:, :(n + 1)], axis=1)
                    pred_rank = pred_raw.argsort().argsort()
                    pred_score, cutoff = utils.getScore(pred_rank, self.cdf_valid, valid=True)
                    kappa_valid = utils.quadratic_weighted_kappa(pred_score, self.Y_valid)
                # 输出文件
                self.out_put_run_fold(run, fold, n, feat_name, trial_counter, kappa_valid, run_fold_matrix.X_train, run_fold_matrix.Y_valid, pred_raw, pred_rank, kappa_cv)

        kappa_cv_mean = np.mean(kappa_cv)
        kappa_cv_std = np.std(kappa_cv)
        if model_param_conf.verbose_level >= 1:
            print("              Mean: %.6f" % kappa_cv_mean)
            print("              Std: %.6f" % kappa_cv_std)
        return kappa_cv_mean, kappa_cv_std

    def predict_all(self, feat_name, trial_counter):
        preds_bagging = np.zeros((self.numTest, self.bagging_size), dtype=float)
        for n in range(model_param_conf.bagging_size):
            pred = self.train_predict(self)
            pred_test = pred
            preds_bagging[:, n] = pred_test

        pred_raw = np.mean(preds_bagging, axis=1)
        pred_rank = pred_raw.argsort().argsort()
