# coding=utf-8
__author__ = 'songquanwang'

import os

import numpy as np

from competition.models.base_model import BaseModel

## sklearn
from sklearn.datasets import dump_svmlight_file
from sklearn.preprocessing import StandardScaler

import competition.conf.model_params_conf as model_param_conf


class LibfmModelImp(BaseModel):
    def __init__(self, param_space, info_folder,feat_folder, feat_name):
        super(LibfmModelImp, self).__init__(param_space,info_folder, feat_folder, feat_name)

    def train_predict(self, matrix, all=False):
        """
        数据训练
        :param train_end_date:
        :return:
        """
        ## scale
        scaler = StandardScaler()
        X_train = matrix.X_train.toarray()
        X_train[matrix.index_base] = scaler.fit_transform(X_train[matrix.index_base])
        ## dump feat
        dump_svmlight_file(X_train[matrix.index_base], matrix.labels_train[matrix.index_base],
                           self.feat_train_path + ".tmp")
        if all:
            X_test = matrix.X_test.toarray()
            X_test = scaler.transform(X_test)
            dump_svmlight_file(X_test, matrix.labels_test, matrix.feat_test_path + ".tmp")

            ## train fm
            cmd = "%s -task r -train %s -test %s -out %s -dim '1,1,%d' -iter %d > libfm.log" % ( \
                model_param_conf.libfm_exe, matrix.feat_train_path + ".tmp", matrix.feat_test_path + ".tmp",
                matrix.raw_pred_test_path, \
                matrix.param['dim'], matrix.param['iter'])
            os.system(cmd)
            os.remove(matrix.feat_train_path + ".tmp")
            os.remove(matrix.feat_test_path + ".tmp")
            ## extract libfm prediction
            pred = np.loadtxt(matrix.raw_pred_test_path, dtype=float)
            ## labels are in [0,1,2,3]
            pred += 1
        else:
            X_valid = matrix.X_valid.toarray()
            X_valid = scaler.transform(X_valid)
            dump_svmlight_file(X_valid, matrix.labels_valid, matrix.feat_valid_path + ".tmp")

            ## train fm
            cmd = "%s -task r -train %s -test %s -out %s -dim '1,1,%d' -iter %d > libfm.log" % ( \
                model_param_conf.libfm_exe, matrix.feat_train_path + ".tmp", matrix.feat_valid_path + ".tmp",
                matrix.raw_pred_valid_path, \
                matrix.param['dim'], matrix.param['iter'])
            os.system(cmd)
            os.remove(matrix.feat_train_path + ".tmp")
            os.remove(matrix.feat_valid_path + ".tmp")
            ## extract libfm prediction
            pred = np.loadtxt(matrix.raw_pred_valid_path, dtype=float)
            ## labels are in [0,1,2,3]
            pred += 1

        return pred

    @staticmethod
    def get_id():
        return "libfm_model_id"

    @staticmethod
    def get_name():
        return "libfm_model"
