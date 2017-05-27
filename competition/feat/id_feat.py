# coding:utf-8
"""
__file__

    genFeat_id_feat.py

__description__

    This file generates the following features for each run and fold, and for the entire training and testing set.

        1. one-hot encoding of query ids (qid)

__author__

    songquanwang

"""

import cPickle

from sklearn.preprocessing import LabelBinarizer

import competition.conf.model_params_conf as config

import abc
from  competition.feat.base_feat import BaseFeat


class IdFeat(BaseFeat):
    __metaclass__ = abc.ABCMeta

    @staticmethod
    def gen_id_feat_run_fold(id_names, run, fold, dfTrain, trainInd, validInd, lb):
        print("Run: %d, Fold: %d" % (run + 1, fold + 1))
        path = "%s/Run%d/Fold%d" % (config.feat_folder, run + 1, fold + 1)
        for id_name in id_names:
            X_train = lb.fit_transform(dfTrain.iloc[trainInd][id_name])
            # 如果validInt 和trainInt没有相同 则transform() X_train没有的classes_会是零向量
            X_valid = lb.transform(dfTrain.iloc[validInd][id_name])
            with open("%s/train.%s.feat.pkl" % (path, id_name), "wb") as f:
                cPickle.dump(X_train, f, -1)
            with open("%s/valid.%s.feat.pkl" % (path, id_name), "wb") as f:
                cPickle.dump(X_valid, f, -1)

    @staticmethod
    def gen_id_feat_all(id_names, dfTrain, dfTest, lb):
        path = "%s/All" % config.feat_folder
        ## use full version for X_train
        for id_name in id_names:
            X_train = lb.fit_transform(dfTrain[id_name])
            X_test = lb.transform(dfTest[id_name])
            with open("%s/train.%s.feat.pkl" % (path, id_name), "wb") as f:
                cPickle.dump(X_train, f, -1)
            with open("%s/test.%s.feat.pkl" % (path, id_name), "wb") as f:
                cPickle.dump(X_test, f, -1)

    @staticmethod
    def gen_id_feat(self):
        """
        入口函数
        :return:
        """
        id_names = ["qid"]
        with open(config.processed_train_data_path, "rb") as f:
            dfTrain = cPickle.load(f)
        with open(config.processed_test_data_path, "rb") as f:
            dfTest = cPickle.load(f)
        ## load pre-defined stratified k-fold index
        with open("%s/stratifiedKFold.%s.pkl" % (config.data_folder, config.stratified_label), "rb") as f:
            skf = cPickle.load(f)

        lb = LabelBinarizer(sparse_output=True)

        print("==================================================")
        print("Generate id features...")

        print("For cross-validation...")
        for run in range(config.n_runs):
            ## use 33% for training and 67 % for validation so we switch trainInd and validInd
            for fold, (validInd, trainInd) in enumerate(skf[run]):
                print("Run: %d, Fold: %d" % (run + 1, fold + 1))
                # 生成 run fold
                self.gen_id_feat_run_fold(id_names, run, fold, dfTrain, trainInd, validInd, lb)

        print("Done.")

        print("For training and testing...")
        self.gen_id_feat_all(id_names, dfTrain, dfTest, lb)
        print("Done.")

        print("All Done.")
