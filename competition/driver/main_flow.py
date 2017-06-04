# coding:utf-8
"""
整体流程
1.预处理
2.生成特征；合并特征
3.生成统计信息文件
4.模型训练、预测结果保存
5.集成训练结果
"""
__author__ = 'songquanwang'
from competition.preprocess.preprocess import preprocess
from competition.preprocess.init_path import init_path
from competition.preprocess.kfold import gen_stratified_kfold
from competition.info.gen_info import gen_info

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

import competition.conf.model_library_config as model_library_config
import competition.models.model_manager as model_manager

from competition.ensemble.predict_ensemble import PredictEnsemble


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
    # 生成数据的统计信息，这些info跟特征无关
    gen_info()


def gen_feat():
    # 不仅生成特征文件，还生成四个特征文件名字的文件 在Feat/solution/counting.feat_name等..
    # 生成所有的特征+label
    # 生成basic tfidf feat
    basic_tfidf_feat = BasicTfidfFeat()
    basic_tfidf_feat.gen_feat_cv()
    # 生成coocrrence tfidf feat
    cooccurence_tfidf_feat = CooccurenceTfidfFeat()
    cooccurence_tfidf_feat.gen_feat_cv()
    # 生成 counting feat
    counting_feat = CountingFeat()
    counting_feat.gen_feat_cv()
    # 生成 distance feat
    distance_feat = DistanceFeat()
    distance_feat.gen_feat_cv()
    id_feat = IdFeat()
    id_feat.gen_feat_cv()

    # 合并所有的feat 生成四个目录，文件名字 train.feat valid.feat test.feat
    BaseFeat.extract_feats_cv(LSA_and_stats_feat_Jun09_Low.feat_names, feat_path_name="LSA_and_stats_feat_Jun09")

    BaseFeat.extract_feats_cv(LSA_svd150_and_Jaccard_coef_Jun14_Low.feat_names, feat_path_name="LSA_svd150_and_Jaccard_coef_Jun14")

    BaseFeat.extract_feats_cv(svd100_and_bow_Jun23_Low.feat_names, feat_path_name="svd100_and_bow_Jun23")

    BaseFeat.extract_feats_cv(svd100_and_bow_Jun27_High.feat_names, feat_path_name="svd100_and_bow_Jun27")


def predict(specified_models):
    """
    使用指定的模型预测结果
    :param specified_models:
    :return:best_kappa_mean, best_kappa_std
    """
    models_best_params = model_manager.make_opt_predict_by_models(specified_models)
    for model_name, best_kappa_mean, best_kappa_std in models_best_params:
        print("Model:%s Mean: %.6f\n Std: %.6f" % (model_name, best_kappa_mean, best_kappa_std))


def ensemble():
    """
    "../../Feat/solution/LSA_and_stats_feat_Jun09"
    :return:
    """
    predict_ensemble = PredictEnsemble()
    feat_folder = model_library_config.feat_folders[0]
    best_kappa_mean, best_kappa_std, best_bagged_model_list, best_bagged_model_weight = predict_ensemble.gen_ensemble(feat_folder)
    print("best_kappa_mean: %.6f\n best_kappa_std: %.6f\n  best_bagged_model_list: %r \n best_bagged_model_weight: %r \n " % (best_kappa_mean, best_kappa_std, best_bagged_model_list, best_bagged_model_weight))

if __name__ == "__main__":
    gen_feat()


