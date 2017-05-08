# -*- coding: utf-8 -*-
__author__ = 'songquanwang'

import abc
import numpy as np
import csv
import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import competition.conf.model_params_conf as model_param_conf
import competition.utils.utils as utils
from scipy.sparse import hstack
from competition.conf.param_config import config
import competition.conf.model_library_config as model_conf


class ModelInter(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, param, feat_folder, feat_name):
        self.param = param
        self.feat_folder = feat_folder
        self.feat_name = feat_name
        self.run_fold_matrix = np.empty((config.n_runs, config.n_folds), dtype=object)
        self.trial_counter = 0
        log_file = "%s/Log/%s_hyperopt.log" % (model_param_conf.output_path, feat_name)
        self.log_handler = open(log_file, 'wb')
        self.writer = csv.writer(self.log_handler)

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
        subm_path = "%s/test.pred.%s_[Id@%d]_[Mean%.6f]_[Std%.6f].csv" % (
            subm_path, feat_name, trial_counter, kappa_cv_mean, kappa_cv_std)

        return raw_pred_test_path, rank_pred_test_path, subm_path

    def get_output_run_fold_path(self, feat_name, trial_counter, run, fold):
        save_path = "%s/Run%d/Fold%d" % (model_param_conf.output_path, run, fold)
        raw_pred_valid_path = "%s/valid.raw.pred.%s_[Id@%d].csv" % (save_path, feat_name, trial_counter)
        rank_pred_valid_path = "%s/valid.pred.%s_[Id@%d].csv" % (save_path, feat_name, trial_counter)

        return raw_pred_valid_path, rank_pred_valid_path

    def feature_all(self):
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

        # 分割训练数据
        index_base, index_meta = utils.bootstrap_all(model_param_conf.bootstrap_replacement, self.numTrain, model_param_conf.bootstrap_ratio)
        self.dtrain = xgb.DMatrix(X_train[index_base], label=labels_train[index_base], weight=self.weight_train[index_base])
        self.dtest = xgb.DMatrix(X_test, label=labels_test)
        # watchlist
        self.watchlist = []
        if model_param_conf.verbose_level >= 2:
            self.watchlist = [(self.dtrain_base, 'train')]

    def feature_run_fold(self, run, fold):
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

        # 分割训练数据
        index_base, index_meta = utils.bootstrap_all(model_param_conf.bootstrap_replacement, matrix.numTrain, model_param_conf.bootstrap_ratio)
        matrix.dtrain = xgb.DMatrix(X_train[index_base], label=labels_train[index_base], weight=matrix.weight_train[index_base])
        matrix.dvalid = xgb.DMatrix(X_valid, label=labels_valid)
        # watchlist
        matrix.watchlist = []
        if model_param_conf.verbose_level >= 2:
            matrix.watchlist = [(matrix.dtrain_base, 'train'), (matrix.dvalid_base, 'valid')]

    def out_put_run_fold(self, run, fold, bagging, feat_name, trial_counter, kappa_valid, X_train, Y_valid, pred_raw,
                         pred_rank, kappa_cv):
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
        :param kappa_cv: out parameter;upate and return
        :return:
        """
        raw_pred_valid_path, rank_pred_valid_path = self.get_output_run_fold_path(feat_name, trial_counter, run, fold)
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

    def out_put_all(self, feat_name, trial_counter, kappa_cv_mean, kappa_cv_std, pred_raw, pred_rank,
                    pred):

        raw_pred_test_path, rank_pred_test_path, subm_path = self.get_output_all_path(feat_name, trial_counter, kappa_cv_mean, kappa_cv_std)
        ## write
        output = pd.DataFrame({"id": self.id_test, "prediction": pred_raw})
        output.to_csv(raw_pred_test_path, index=False)

        ## write
        output = pd.DataFrame({"id": self.id_test, "prediction": pred_rank})
        output.to_csv(rank_pred_test_path, index=False)

        ## write score
        pred_score = utils.getScore(pred, self.cdf_test)
        output = pd.DataFrame({"id": self.id_test, "prediction": pred_score})
        output.to_csv(subm_path, index=False)

    def hyperopt_obj(self, feat_folder, feat_name, trial_counter):
        kappa_cv = np.zeros((config.n_runs, config.n_folds), dtype=float)
        for run in range(1, config.n_runs + 1):
            for fold in range(1, config.n_folds + 1):
                # 生成 run_fold_matrix
                matrix = self.run_fold_matrix[run][fold]
                self.feature_run_fold(self, run, fold, matrix)
                preds_bagging = np.zeros((matrix.numValid, model_param_conf.bagging_size), dtype=float)
                for n in range(model_param_conf.bagging_size):
                    pred = self.train_predict(matrix)
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
                self.out_put_run_fold(run, fold, n, feat_name, trial_counter, kappa_valid, matrix.X_train,
                                      matrix.Y_valid, pred_raw, pred_rank, kappa_cv)

        kappa_cv_mean = np.mean(kappa_cv)
        kappa_cv_std = np.std(kappa_cv)
        if model_param_conf.verbose_level >= 1:
            print("              Mean: %.6f" % kappa_cv_mean)
            print("              Std: %.6f" % kappa_cv_std)
        # 生成提交结果
        self.predict_all(self, feat_folder, feat_name, trial_counter, kappa_cv_mean, kappa_cv_std)
        return kappa_cv_mean, kappa_cv_std

    def hyperopt_wrapper(self, param, feat_folder, feat_name):

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

        ## evaluate performance
        kappa_cv_mean, kappa_cv_std = self.hyperopt_obj(param, feat_folder, feat_name, self.trial_counter)

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

        return {'loss': -kappa_cv_mean, 'attachments': {'std': kappa_cv_std}, 'status': STATUS_OK}

    def predict_all(self, feat_folder, feat_name, trial_counter, kappa_cv_mean, kappa_cv_std):
        preds_bagging = np.zeros((self.numTest, self.bagging_size), dtype=float)
        for n in range(model_param_conf.bagging_size):
            pred = self.train_predict(self)
            pred_test = pred
            preds_bagging[:, n] = pred_test

        pred_raw = np.mean(preds_bagging, axis=1)
        pred_rank = pred_raw.argsort().argsort()
        self.out_put_all(feat_folder, feat_name, trial_counter, kappa_cv_mean, kappa_cv_std, pred_raw, pred_rank,
                         pred)

    @abc.abstractmethod
    def get_id(self):
        return

    @abc.abstractmethod
    def get_name(self):
        return

    def pre_process(self):
        return

    def optmize(self, feat_folder, feat_name):
        param_space = model_conf.param_spaces[feat_name]
        # 记录日志 到output/***_hyperopt.log

        # 每行日志都包含 'trial_counter', 'kappa_mean', 'kappa_std' 三个字段 + 模型参数
        headers = ['trial_counter', 'kappa_mean', 'kappa_std']
        for k, v in sorted(param_space.items()):
            headers.append(k)
        self.writer.writerow(headers)
        self.log_handler.flush()

        print("************************************************************")
        print("Search for the best params")
        # global trial_counter
        trials = Trials()
        objective = lambda p: self.hyperopt_wrapper(p, feat_folder, feat_name)
        best_params = fmin(objective, param_space, algo=tpe.suggest,
                           trials=trials, max_evals=param_space["max_evals"])
        # 把best_params包含的数字属性转成int
        for f in model_conf.int_feat:
            if best_params.has_key(f):
                best_params[f] = int(best_params[f])
        print("************************************************************")
        print("Best params")
        for k, v in best_params.items():
            print "        %s: %s" % (k, v)
        # 获取尝试的losses
        trial_kappas = -np.asarray(trials.losses(), dtype=float)
        best_kappa_mean = max(trial_kappas)
        # where返回两个维度的坐标
        ind = np.where(trial_kappas == best_kappa_mean)[0][0]
        # 找到最优参数的std
        best_kappa_std = trials.trial_attachments(trials.trials[ind])['std']
        print("Kappa stats")
        print("        Mean: %.6f\n        Std: %.6f" % (best_kappa_mean, best_kappa_std))
        return best_kappa_mean, best_kappa_std
