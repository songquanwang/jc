# coding:utf-8
__author__ = 'songquanwang'
from competition.preprocess.preprocess import init_path, preprocess, gen_stratified_kfold
from competition.info.gen_info import gen_info


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
    return


def predict():
    return


def ensemble():
    return
