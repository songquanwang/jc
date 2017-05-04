# coding=utf-8
__author__ = 'songquanwang'

import numpy as np
from sklearn.preprocessing import StandardScaler

from competition.inter.model_inter import ModelInter
## keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
## cutomized module
from competition.conf.param_config import config

global trial_counter
global log_handler


class GbdtModelImp(ModelInter):
    def __init__(self, param, feat_folder, feat_name):
        super(ModelInter, self).__init__(param, feat_folder, feat_name)

    def train_predict(self, matrix, all=False):
        """
        数据训练
        :param train_end_date:
        :return:
        """
        param = matrix.param
        ## regression with keras' deep neural networks
        model = Sequential()
        ## input layer
        model.add(Dropout(param["input_dropout"]))
        ## hidden layers
        first = True
        hidden_layers = param['hidden_layers']
        ## scale
        scaler = StandardScaler()
        X_train = matrix.X_train.toarray()
        X_train[matrix.index_base] = scaler.fit_transform(X_train[matrix.index_base])
        if all:
            while hidden_layers > 0:
                if first:
                    dim = X_train.shape[1]
                    first = False
                else:
                    dim = param["hidden_units"]
                model.add(Dense(dim, param["hidden_units"], init='glorot_uniform'))
                if param["batch_norm"]:
                    model.add(BatchNormalization((param["hidden_units"],)))
                if param["hidden_activation"] == "prelu":
                    model.add(PReLU((param["hidden_units"],)))
                else:
                    model.add(Activation(param['hidden_activation']))
                model.add(Dropout(param["hidden_dropout"]))
                hidden_layers -= 1
            ## output layer
            model.add(Dense(param["hidden_units"], 1, init='glorot_uniform'))
            model.add(Activation('linear'))
            ## loss
            model.compile(loss='mean_squared_error', optimizer="adam")
            ## to array
            X_test = matrix.X_test.toarray()
            X_test = scaler.transform(X_test)
            ## train
            model.fit(X_train[matrix.index_base], matrix.labels_train[matrix.index_base] + 1,
                      nb_epoch=param['nb_epoch'], batch_size=param['batch_size'],
                      testation_split=0, verbose=0)
            ##prediction
            pred = model.predict(X_test, verbose=0)
            pred.shape = (X_test.shape[0],)

        else:
            while hidden_layers > 0:
                if first:
                    dim = X_train.shape[1]
                    first = False
                else:
                    dim = param["hidden_units"]
                model.add(Dense(dim, param["hidden_units"], init='glorot_uniform'))
                if param["batch_norm"]:
                    model.add(BatchNormalization((param["hidden_units"],)))
                if param["hidden_activation"] == "prelu":
                    model.add(PReLU((param["hidden_units"],)))
                else:
                    model.add(Activation(param['hidden_activation']))
                model.add(Dropout(param["hidden_dropout"]))
                hidden_layers -= 1
            ## output layer
            model.add(Dense(param["hidden_units"], 1, init='glorot_uniform'))
            model.add(Activation('linear'))
            ## loss
            model.compile(loss='mean_squared_error', optimizer="adam")
            ## to array
            X_valid = matrix.X_valid.toarray()
            X_valid = scaler.transform(X_valid)
            ## train
            model.fit(X_train[matrix.index_base], matrix.labels_train[matrix.index_base] + 1,
                      nb_epoch=param['nb_epoch'], batch_size=param['batch_size'],
                      validation_split=0, verbose=0)
            ##prediction
            pred = model.predict(X_valid, verbose=0)
            pred.shape = (X_valid.shape[0],)

        return pred

    def get_predicts(self):
        return

    @staticmethod
    def get_id():
        return "gdbt_model_id"

    @staticmethod
    def get_name():
        return "gdbt_model"
