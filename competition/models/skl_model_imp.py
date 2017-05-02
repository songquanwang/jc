# coding=utf-8
__author__ = 'songquanwang'

import numpy as np
import pandas as pd

from competition.inter.model_inter import ModelInter

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.linear_model import Ridge, Lasso, LassoLars, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
# 梯度自举树，也是gdbt的实现
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
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

    def train_predict(self, set_obj):
        """
        数据训练
        :param train_end_date:
        :return:
        """

        if self.param['task'] == "reg_skl_rf":
            pred = self.reg_skl_rf_predict(self, set_obj)

        elif self.param['task'] == "reg_skl_etr":

            pred = self.reg_skl_etr_predict(self, set_obj)
        elif self.param['task'] == "reg_skl_gbm":

            pred = self.reg_skl_gbm_predict(self, set_obj)
        elif self.param['task'] == "clf_skl_lr":

            pred = self.clf_skl_lr_predict(self, set_obj)
        elif self.param['task'] == "reg_skl_svr":
            pred = self.reg_skl_svr_predict(self, set_obj)

        elif self.param['task'] == "reg_skl_ridge":
            pred = self.reg_skl_ridge_predict(self, set_obj)

        elif self.param['task'] == "reg_skl_lasso":
            pred = self.reg_skl_lasso_predict(self, set_obj)
        return pred

    def reg_skl_rf_predict(self, set_obj):
        ## regression with sklearn random forest regressor
        param = self.param
        rf = RandomForestRegressor(n_estimators=param['n_estimators'],
                                   max_features=param['max_features'],
                                   n_jobs=param['n_jobs'],
                                   random_state=param['random_state'])
        rf.fit(set_obj.X_train[set_obj.index_base], set_obj.labels_train[set_obj.index_base] + 1,
               sample_weight=set_obj.weight_train[set_obj.index_base])
        pred = rf.predict(set_obj.X_valid)
        return pred

    def reg_skl_etr_predict(self, set_obj):
        ## regression with sklearn extra trees regressor
        param = self.param
        etr = ExtraTreesRegressor(n_estimators=param['n_estimators'],
                                  max_features=param['max_features'],
                                  n_jobs=param['n_jobs'],
                                  random_state=param['random_state'])
        etr.fit(set_obj.X_train[set_obj.index_base], set_obj.labels_train[set_obj.index_base] + 1,
                sample_weight=set_obj.weight_train[set_obj.index_base])
        pred = etr.predict(set_obj.X_valid)
        return pred

    def reg_skl_gbm_predict(self, set_obj):
        ## regression with sklearn gradient boosting regressor
        param = self.param

        gbm = GradientBoostingRegressor(n_estimators=param['n_estimators'],
                                        max_features=param['max_features'],
                                        learning_rate=param['learning_rate'],
                                        max_depth=param['max_depth'],
                                        subsample=param['subsample'],
                                        random_state=param['random_state'])
        gbm.fit(set_obj.X_train.toarray()[set_obj.index_base], set_obj.labels_train[set_obj.index_base] + 1,
                sample_weight=set_obj.weight_train[set_obj.index_base])
        pred = gbm.predict(set_obj.X_valid.toarray())
        return pred

    def clf_skl_lr_predict(self, set_obj):
        ## classification with sklearn logistic regression   只寻找一个参数的最优参数
        param = self.param
        lr = LogisticRegression(penalty="l2", dual=True, tol=1e-5,
                                C=param['C'], fit_intercept=True, intercept_scaling=1.0,
                                class_weight='auto', random_state=param['random_state'])
        lr.fit(set_obj.X_train[set_obj.index_base], set_obj.labels_train[set_obj.index_base] + 1)
        pred = lr.predict_proba(set_obj.X_valid)
        w = np.asarray(range(1, model_param_conf.numOfClass + 1))
        pred = pred * w[np.newaxis, :]
        pred = np.sum(pred, axis=1)
        return pred

    def reg_skl_svr_predict(self, set_obj):
        ## regression with sklearn support vector regression
        param = self.param
        X_train, X_valid = set_obj.X_train.toarray(), set_obj.X_valid.toarray()
        scaler = StandardScaler()
        X_train[set_obj.index_base] = scaler.fit_transform(set_obj.X_train[set_obj.index_base])
        X_valid = scaler.transform(X_valid)
        svr = SVR(C=param['C'], gamma=param['gamma'], epsilon=param['epsilon'],
                  degree=param['degree'], kernel=param['kernel'])
        svr.fit(X_train[set_obj.index_base], set_obj.labels_train[set_obj.index_base] + 1,
                sample_weight=set_obj.weight_train[set_obj.index_base])
        pred = svr.predict(X_valid)
        return pred

    def reg_skl_ridge_predict(self, set_obj):
        ## regression with sklearn ridge regression
        param = self.param
        ridge = Ridge(alpha=param["alpha"], normalize=True)
        ridge.fit(set_obj.X_train[set_obj.index_base], set_obj.labels_train[set_obj.index_base] + 1,
                  sample_weight=set_obj.weight_train[set_obj.index_base])
        pred = ridge.predict(set_obj.X_valid)
        return pred

    def reg_skl_lasso_predict(self, set_obj):
        ## regression with sklearn lasso
        param = self.param
        lasso = Lasso(alpha=param["alpha"], normalize=True)
        lasso.fit(set_obj.X_train[set_obj.index_base], set_obj.labels_train[set_obj.index_base] + 1)
        pred = lasso.predict(set_obj.X_valid)

        return pred

    def get_predicts(self):
        return

    @staticmethod
    def get_id():
        return "gdbt_model_id"

    @staticmethod
    def get_name():
        return "gdbt_model"
