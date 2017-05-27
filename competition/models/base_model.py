# -*- coding: utf-8 -*-
__author__ = 'songquanwang'

import abc
import csv

import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
import xgboost as xgb
from hyperopt import STATUS_OK
from scipy.sparse import hstack

import competition.conf.model_params_conf as model_param_conf
import competition.utils.utils as utils
import competition.conf.model_library_config as config
import competition.conf.model_library_config as model_conf


class BaseModel(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, param_space, feat_folder, feat_name):
        self.param_space = param_space
        self.feat_folder = feat_folder
        self.feat_name = feat_name
        self.run_fold_matrix = np.empty((config.n_runs, config.n_folds), dtype=object)
        self.trial_counter = 0
        log_file = "%s/Log/%s_hyperopt.log" % (model_param_conf.output_path, feat_name)
        self.log_handler = open(log_file, 'wb')
        self.writer = csv.writer(self.log_handler)

    def init_all_path(self):
        path = "%s/All" % (self.feat_folder)
        self.feat_train_path = "%s/train.feat" % path
        self.feat_test_path = "%s/test.feat" % path

        self.weight_train_path = "%s/train.feat.weight" % path

        self.info_train_path = "%s/train.info" % path
        self.info_test_path = "%s/test.info" % path

        self.cdf_test_path = "%s/test.cdf" % path

    def init_run_fold_path(self, run, fold, matrix):
        path = "%s/Run%d/Fold%d" % (self.feat_folder, run, fold)
        matrix.feat_train_path = "%s/train.feat" % path
        matrix.feat_valid_path = "%s/valid.feat" % path

        matrix.weight_train_path = "%s/train.feat.weight" % path
        matrix.weight_valid_path = "%s/valid.feat.weight" % path

        matrix.info_train_path = "%s/train.info" % path
        matrix.info_valid_path = "%s/valid.info" % path

        matrix.cdf_valid_path = "%s/valid.cdf" % path

    def get_output_all_path(self, feat_name, trial_counter, kappa_cv_mean, kappa_cv_std):
        save_path = "%s/All" % model_param_conf.output_path
        subm_path = "%s/Subm" % model_param_conf.output_path
        raw_pred_test_path = "%s/test.raw.pred.%s_[Id@%d].csv" % (save_path, feat_name, trial_counter)
        rank_pred_test_path = "%s/test.pred.%s_[Id@%d].csv" % (save_path, feat_name, trial_counter)
        # submission path (relevance as in [1,2,3,4])
        subm_path = "%s/test.pred.%s_[Id@%d]_[Mean%.6f]_[Std%.6f].csv" % (subm_path, feat_name, trial_counter, kappa_cv_mean, kappa_cv_std)

        return raw_pred_test_path, rank_pred_test_path, subm_path

    def get_output_run_fold_path(self, feat_name, trial_counter, run, fold):
        save_path = "%s/Run%d/Fold%d" % (model_param_conf.output_path, run, fold)
        raw_pred_valid_path = "%s/valid.raw.pred.%s_[Id@%d].csv" % (save_path, feat_name, trial_counter)
        rank_pred_valid_path = "%s/valid.pred.%s_[Id@%d].csv" % (save_path, feat_name, trial_counter)

        return raw_pred_valid_path, rank_pred_valid_path

    def gen_set_obj_all(self):
        # init the path
        self.init_all_path()
        # feat
        X_train, labels_train = load_svmlight_file(self.feat_train_path)
        X_test, labels_test = load_svmlight_file(self.feat_test_path)
        # 延展array
        if X_test.shape[1] < X_train.shape[1]:
            X_test = hstack([X_test, np.zeros((X_test.shape[0], X_train.shape[1] - X_test.shape[1]))])
        elif X_test.shape[1] > X_train.shape[1]:
            X_train = hstack([X_train, np.zeros((X_train.shape[0], X_test.shape[1] - X_train.shape[1]))])
        X_train = X_train.tocsr()
        X_test = X_test.tocsr()
        # 赋给成员变量
        self.X_train, self.labels_train, self.X_test, self.labels_test = X_train, labels_train, X_test, labels_test
        # weight
        self.weight_train = np.loadtxt(self.weight_train_path, dtype=float)
        # info
        self.info_train = pd.read_csv(self.info_train_path)
        self.info_test = pd.read_csv(self.info_test_path)
        # cdf
        self.cdf_test = np.loadtxt(self.cdf_test_path, dtype=float)
        # number
        self.numTrain = self.info_train.shape[0]
        self.numTest = self.info_test.shape[0]

        # 对数据进行自举法抽样；因为ratio=1 且bootstrap_replacement=false 说明没有用到，就使用的是全量数据
        index_base, index_meta = utils.bootstrap_all(model_param_conf.bootstrap_replacement, self.numTrain, model_param_conf.bootstrap_ratio)
        self.dtrain = xgb.DMatrix(X_train[index_base], label=labels_train[index_base], weight=self.weight_train[index_base])
        self.dtest = xgb.DMatrix(X_test, label=labels_test)
        # watchlist
        self.watchlist = []
        if model_param_conf.verbose_level >= 2:
            self.watchlist = [(self.dtrain_base, 'train')]

    def gen_set_obj_run_fold(self, run, fold):
        """
        每个run 每个fold 生成
        :param run:
        :param fold:
        :return:
        """
        # init the path
        self.init_run_fold_path(self, run, fold)
        matrix = self.run_fold_matrix[run][fold]

        # feat
        X_train, labels_train = load_svmlight_file(matrix.feat_train_path)
        X_valid, labels_valid = load_svmlight_file(matrix.feat_valid_path)
        # 延展array
        if X_valid.shape[1] < X_train.shape[1]:
            X_valid = hstack([X_valid, np.zeros((X_valid.shape[0], X_train.shape[1] - X_valid.shape[1]))])
        elif X_valid.shape[1] > X_train.shape[1]:
            X_train = hstack([X_train, np.zeros((X_train.shape[0], X_valid.shape[1] - X_train.shape[1]))])
        X_train = X_train.tocsr()
        X_valid = X_valid.tocsr()
        # 赋给成员变量
        matrix.X_train, matrix.labels_train, matrix.X_valid, matrix.labels_valid = X_train, labels_train, X_valid, labels_valid
        # weight
        matrix.weight_train = np.loadtxt(matrix.weight_train_path, dtype=float)
        matrix.weight_valid = np.loadtxt(matrix.weight_valid_path, dtype=float)
        # info
        matrix.info_train = pd.read_csv(matrix.info_train_path)
        matrix.info_valid = pd.read_csv(matrix.info_valid_path)
        # cdf
        matrix.cdf_valid = np.loadtxt(matrix.cdf_valid_path, dtype=float)
        # number
        matrix.numTrain = matrix.info_train.shape[0]
        matrix.numValid = matrix.info_valid.shape[0]

        # 对数据进行自举法抽样；因为ratio=1 且bootstrap_replacement=false 说明没有用到，就使用的是全量数据
        index_base, index_meta = utils.bootstrap_all(model_param_conf.bootstrap_replacement, matrix.numTrain, model_param_conf.bootstrap_ratio)
        matrix.dtrain = xgb.DMatrix(X_train[index_base], label=labels_train[index_base], weight=matrix.weight_train[index_base])
        matrix.dvalid = xgb.DMatrix(X_valid, label=labels_valid)
        # watchlist
        matrix.watchlist = []
        if model_param_conf.verbose_level >= 2:
            matrix.watchlist = [(matrix.dtrain_base, 'train'), (matrix.dvalid_base, 'valid')]
        return matrix

    def out_put_run_fold(self, run, fold, feat_name, trial_counter, X_train, Y_valid, pred_raw, pred_rank, kappa_valid):
        """

        :param run:
        :param fold:
        :param bagging:
        :param feat_name:
        :param trial_counter:
        :param kappa_valid:
        :param X_train:
        :param Y_valid:
        :param pred_raw:
        :param pred_rank:
        :return:
        """
        raw_pred_valid_path, rank_pred_valid_path = self.get_output_run_fold_path(feat_name, trial_counter, run, fold)
        # save this prediction
        dfPred = pd.DataFrame({"target": Y_valid, "prediction": pred_raw})
        dfPred.to_csv(raw_pred_valid_path, index=False, header=True, columns=["target", "prediction"])
        # save this prediction
        dfPred = pd.DataFrame({"target": Y_valid, "prediction": pred_rank})
        dfPred.to_csv(rank_pred_valid_path, index=False, header=True, columns=["target", "prediction"])

    def out_put_all(self, feat_name, trial_counter, kappa_cv_mean, kappa_cv_std, pred_raw, pred_rank):

        raw_pred_test_path, rank_pred_test_path, subm_path = self.get_output_all_path(feat_name, trial_counter, kappa_cv_mean, kappa_cv_std)
        ## write
        output = pd.DataFrame({"id": self.id_test, "prediction": pred_raw})
        output.to_csv(raw_pred_test_path, index=False)

        ## write
        output = pd.DataFrame({"id": self.id_test, "prediction": pred_rank})
        output.to_csv(rank_pred_test_path, index=False)

        ## write score pred--原来代码有错：应该是pred_raw 因为pred_raw是多次装袋后平均预测值，不应该是其中一次装袋的预测值
        pred_score = utils.getScore(pred_raw, self.cdf_test)
        output = pd.DataFrame({"id": self.id_test, "prediction": pred_score})
        output.to_csv(subm_path, index=False)

    def gen_bagging(self, set_obj, all):
        """
        分袋整合预测结果
        :param set_obj:
        :param all:
        :return:
        """
        preds_bagging = np.zeros((self.numTest, model_param_conf.bagging_size), dtype=float)
        for n in range(model_param_conf.bagging_size):
            # 调用 每个子类的train_predict方法，多态
            pred = self.train_predict(self, set_obj, all)
            pred_test = pred
            preds_bagging[:, n] = pred_test
            if not all:
                # 每次会把当前bagging的结果累计进来 求均值
                pred_raw = np.mean(preds_bagging[:, :(pred + 1)], axis=1)
                # 为什么需要两次argsort？
                pred_rank = pred_raw.argsort().argsort()
                pred_score, cutoff = utils.getScore(pred_rank, self.cdf_valid, valid=True)
                kappa_valid = utils.quadratic_weighted_kappa(pred_score, self.Y_valid)

        pred_raw = np.mean(preds_bagging, axis=1)
        pred_rank = pred_raw.argsort().argsort()
        if all:
            return pred_raw, pred_rank
        else:
            return pred_raw, pred_rank, kappa_valid

    def hyperopt_obj(self, param, feat_folder, feat_name, trial_counter):
        """
        最优化方法 hyperopt_obj
        :param feat_folder:
        :param feat_name:
        :param trial_counter:
        :return:
        """
        # 定义kappa交叉验证结构
        kappa_cv = np.zeros((config.n_runs, config.n_folds), dtype=float)
        for run in range(1, config.n_runs + 1):
            for fold in range(1, config.n_folds + 1):
                # 生成 run_fold_set_obj
                set_obj = self.gen_set_obj_run_fold(self, run, fold)
                # bagging结果
                pred_raw, pred_rank, kappa_valid = self.gen_bagging(self, set_obj, all=False)
                # 输出文件
                kappa_cv[run - 1, fold - 1] = kappa_valid
                # 生成没run fold的结果
                self.out_put_run_fold(run, fold, feat_name, trial_counter, set_obj.X_train, set_obj.Y_valid, pred_raw, pred_rank, kappa_valid)
        # kappa_cv run*fold*bagging_size 均值和方差
        kappa_cv_mean, kappa_cv_std = np.mean(kappa_cv), np.std(kappa_cv)
        if model_param_conf.verbose_level >= 1:
            print("              Mean: %.6f" % kappa_cv_mean)
            print("              Std: %.6f" % kappa_cv_std)
        # # bagging结果
        pred_raw, pred_rank = self.gen_bagging(self, set_obj, all=True)
        # 生成提交结果
        self.out_put_all(feat_folder, feat_name, trial_counter, kappa_cv_mean, kappa_cv_std, pred_raw, pred_rank)
        # 记录参数文件
        self.log_param(param, feat_name, kappa_cv_mean, kappa_cv_std)
        # 根据交叉验证的平均值作为模型好坏标准
        return {'loss': -kappa_cv_mean, 'attachments': {'std': kappa_cv_std}, 'status': STATUS_OK}

    def log_param(self, param, feat_name, kappa_cv_mean, kappa_cv_std):
        """
        记录参数文件
        :param param:
        :param feat_name:
        :param kappa_cv_mean:
        :param kappa_cv_std:
        :return:
        """
        self.trial_counter += 1
        # convert integer feat
        for f in model_conf.int_feat:
            if param.has_key(f):
                param[f] = int(param[f])

        print("------------------------------------------------------------")
        print "Trial %d" % self.trial_counter

        print("        Model")
        print("              %s" % feat_name)
        print("        Param")
        for k, v in sorted(param.items()):
            print("              %s: %s" % (k, v))
        print("        Result")
        print("                    Run      Fold      Bag      Kappa      Shape")
        ## log
        var_to_log = [
            "%d" % self.trial_counter,
            "%.6f" % kappa_cv_mean,
            "%.6f" % kappa_cv_std
        ]
        # 日志中输出参数值 次数 均值 标准差 值1 值2 ...值N
        for k, v in sorted(param.items()):
            var_to_log.append("%s" % v)
        self.writer.writerow(var_to_log)
        self.log_handler.flush()

    def log_header(self):
        """
        log 记录文件头部
        :return:
        """
        # 记录日志 到output/***_hyperopt.log
        # 每行日志都包含 'trial_counter', 'kappa_mean', 'kappa_std' 三个字段 + 模型参数
        headers = ['trial_counter', 'kappa_mean', 'kappa_std']
        for k, v in sorted(self.param_space.items()):
            headers.append(k)
        self.writer.writerow(headers)
        self.log_handler.flush()

    @abc.abstractmethod
    def train_predict(self, matrix, all=False):
        """
        所有子类模型都需要实现这个方法
        :param matrix:
        :param all:
        :return:
        """
        return



