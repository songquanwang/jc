# coding:utf-8
__author__ = 'songquanwang'
from competition.preprocess.preprocess import init_path, preprocess, gen_stratified_kfold
from competition.info.gen_info import gen_info
import competition.models.model_manager as model_manager
from hyperopt import fmin, tpe, STATUS_OK, Trials

import numpy as np
import os
import competition.conf.model_params_conf as config
import competition.conf.model_library_config  as model_library_config

import competition.feat.conf.LSA_and_stats_feat_Jun09_Low as LSA_and_stats_feat_Jun09_Low
import competition.feat.conf.LSA_svd150_and_Jaccard_coef_Jun14_Low as LSA_svd150_and_Jaccard_coef_Jun14_Low
import competition.feat.conf.svd100_and_bow_Jun23_Low as svd100_and_bow_Jun23_Low
import competition.feat.conf.svd100_and_bow_Jun27_High as svd100_and_bow_Jun27_High

from competition.feat.base_feat import BaseModel


def preprocess():
    """
    预处理
     1.构建必要的目录
     2.预处理
     3.交叉验证
    :return:
    """
    init_path()
    preprocess()
    gen_stratified_kfold()


def gen_info():
    """
    创建info文件
    :return:
    """
    gen_info(feat_path_name="LSA_and_stats_feat_Jun09")

    gen_info(feat_path_name="LSA_svd150_and_Jaccard_coef_Jun14")

    gen_info(feat_path_name="svd100_and_bow_Jun23")

    gen_info(feat_path_name="svd100_and_bow_Jun27")


def gen_feat():

    # 生成所有的特征+label

    BaseModel.combine_feat(LSA_and_stats_feat_Jun09_Low.feat_names, feat_path_name="LSA_and_stats_feat_Jun09")

    BaseModel.combine_feat(LSA_svd150_and_Jaccard_coef_Jun14_Low.feat_names, feat_path_name="LSA_svd150_and_Jaccard_coef_Jun14")

    BaseModel.combine_feat(svd100_and_bow_Jun23_Low.feat_names, feat_path_name="svd100_and_bow_Jun23")

    BaseModel.combine_feat(svd100_and_bow_Jun27_High.feat_names, feat_path_name="svd100_and_bow_Jun27")


def predict(specified_models):
    """
    使用指定的模型预测结果
    :param specified_models:
    :return:best_kappa_mean, best_kappa_std
    """
    best_kappa_mean, best_kappa_std = model_manager.make_predict_by_models(specified_models)
    print("Mean: %.6f\n Std: %.6f" % (best_kappa_mean, best_kappa_std))


def ensemble():
    return
