# coding=utf-8
__author__ = 'songquanwang'

import numpy as np
import pandas as pd

from competition.inter.model_inter import ModelInter
import xgboost as xgb
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
        bst = xgb.train(self.param, set_obj.dtrain_base, self.param['num_round'], set_obj.watchlist,
                        feval=evalerror_regrank_valid)
        pred = bst.predict(set_obj.dvalid_base)
        return pred

    def soft_max_predict(self, set_obj):
        evalerror_softmax_valid = lambda preds, dtrain: utils.evalerror_softmax_cdf(preds, dtrain, set_obj.cdf_valid)
        ## softmax regression with xgboost
        bst = xgb.train(self.param, set_obj.dtrain_base, self.param['num_round'], set_obj.watchlist,
                        feval=evalerror_softmax_valid)
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
        bst = xgb.train(self.param, set_obj.dtrain_base, self.param['num_round'], set_obj.watchlist, obj=obj,
                        feval=utils.evalerror_softkappa_valid)
        pred = utils.softmax(bst.predict(set_obj.dvalid_base))
        w = np.asarray(range(1, model_param_conf.num_of_class + 1))
        pred = pred * w[np.newaxis, :]
        pred = np.sum(pred, axis=1)
        return pred

    def ebc_predict(self, set_obj):
        # ebc with xgboost
        obj = lambda preds, dtrain: utils.ebcObj(preds, dtrain)
        bst = xgb.train(self.param, set_obj.dtrain_base, self.param['num_round'], set_obj.watchlist, obj=obj,
                        feval=utils.evalerror_ebc_valid)
        pred = utils.sigmoid(bst.predict(set_obj.dvalid_base))
        pred = utils.applyEBCRule(pred, hard_threshold=utils.ebc_hard_threshold)
        return pred

    def cocr_predict(self, set_obj):
        ## cocr with xgboost
        obj = lambda preds, dtrain: utils.cocrObj(preds, dtrain)
        bst = xgb.train(self.param, set_obj.dtrain_base, self.param['num_round'], set_obj.watchlist, obj=obj,
                        feval=utils.evalerror_cocr_valid)
        pred = bst.predict(set_obj.dvalid_base)
        pred = utils.applyCOCRRule(pred)
        return pred

    def get_predicts(self):
        return

    @staticmethod
    def get_id():
        return "gdbt_model_id"

    @staticmethod
    def get_name():
        return "gdbt_model"
