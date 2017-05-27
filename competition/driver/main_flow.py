# coding:utf-8
__author__ = 'songquanwang'
from competition.preprocess.preprocess import preprocess
from competition.preprocess.init_path import init_path
from competition.preprocess.kfold import gen_stratified_kfold
from competition.info.gen_info import gen_info
import competition.models.model_manager as model_manager

import competition.conf.feat.LSA_and_stats_feat_Jun09_Low as LSA_and_stats_feat_Jun09_Low
import competition.conf.feat.LSA_svd150_and_Jaccard_coef_Jun14_Low as LSA_svd150_and_Jaccard_coef_Jun14_Low
import competition.conf.feat.svd100_and_bow_Jun23_Low as svd100_and_bow_Jun23_Low
import competition.conf.feat.svd100_and_bow_Jun27_High as svd100_and_bow_Jun27_High

from competition.feat.base_feat import BaseFeat
from competition.feat.basic_tfidf_feat import BasicTfidfFeat
from competition.feat.cooccurrence_tfidf_feat import CooccurenceTfidfFeat
from competition.feat.counting_feat import CountingFeat
from competition.feat.distance_feat import DistanceFeat
from competition.feat.id_feat import IdFeat
import competition.conf.feat_params_conf as feat_param_conf
from competition.ensemble.predict_ensemble import PredictEnsemble
import competition.conf.model_library_config as model_library_config


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
    stats_feat_flag = feat_param_conf.stats_feat_flag
    # 生成basic tfidf feat
    basic_tfidf_feat = BasicTfidfFeat(stats_feat_flag)
    basic_tfidf_feat.gen_basic_tfidf_feat()
    # 生成coocrrence tfidf feat
    cooccurence_tfidf_feat = CooccurenceTfidfFeat()
    cooccurence_tfidf_feat.gen_coocurrence_tfidf_feat()
    # 生成 counting feat
    counting_feat = CountingFeat()
    counting_feat.gen_counting_feat()
    # 生成 distance feat
    distance_feat = DistanceFeat(stats_feat_flag)
    distance_feat.gen_distance_feat()
    # 生成id feat
    id_feat = IdFeat()
    id_feat.gen_id_feat()

    # 合并所有的feat 生成四个目录，文件名字 train.feat valid.feat test.feat
    BaseFeat.combine_feat(LSA_and_stats_feat_Jun09_Low.feat_names, feat_path_name="LSA_and_stats_feat_Jun09")

    BaseFeat.combine_feat(LSA_svd150_and_Jaccard_coef_Jun14_Low.feat_names,
                          feat_path_name="LSA_svd150_and_Jaccard_coef_Jun14")

    BaseFeat.combine_feat(svd100_and_bow_Jun23_Low.feat_names, feat_path_name="svd100_and_bow_Jun23")

    BaseFeat.combine_feat(svd100_and_bow_Jun27_High.feat_names, feat_path_name="svd100_and_bow_Jun27")


def predict(specified_models):
    """
    使用指定的模型预测结果
    :param specified_models:
    :return:best_kappa_mean, best_kappa_std
    """
    best_kappa_mean, best_kappa_std = model_manager.make_predict_by_models(specified_models)
    print("Mean: %.6f\n Std: %.6f" % (best_kappa_mean, best_kappa_std))


def ensemble():
    """
    "../../Feat/solution/LSA_and_stats_feat_Jun09"
    :return:
    """
    predict_ensemble =PredictEnsemble()
    feat_folder =model_library_config.feat_folders[0]
    best_kappa_mean, best_kappa_std, best_bagged_model_list, best_bagged_model_weight =predict_ensemble.gen_ensemble(feat_folder)
    print("best_kappa_mean: %.6f\n best_kappa_std: %.6f\n  best_bagged_model_list: %r \n best_bagged_model_weight: %r \n " % (best_kappa_mean, best_kappa_std, best_bagged_model_list, best_bagged_model_weight))
