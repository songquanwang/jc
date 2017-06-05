# coding=utf-8
__author__ = 'songquanwang'

from sklearn.preprocessing import StandardScaler

from competition.models.base_model import BaseModel
## keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU


class KerasDnnModelImp(BaseModel):
    def __init__(self, param_space, info_folder,feat_folder, feat_name):
        super(KerasDnnModelImp, self).__init__(param_space, info_folder,feat_folder, feat_name)

    def train_predict(self,param, set_obj, all=False):
        """
        数据训练
        :param train_end_date:
        :return:
        """
        # regression with keras' deep neural networks
        model = Sequential()
        # input layer
        model.add(Dropout(param["input_dropout"]))
        # hidden layers
        first = True
        hidden_layers = param['hidden_layers']
        # scale
        scaler = StandardScaler()
        X_train = set_obj.X_train.toarray()
        X_train[set_obj.index_base] = scaler.fit_transform(X_train[set_obj.index_base])
        # test or valid to array
        if all:
            # to array
            X_test = scaler.transform(set_obj.X_test.toarray())
        else:
            X_test = scaler.transform(set_obj.X_valid.toarray())

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
        # output layer
        model.add(Dense(param["hidden_units"], 1, init='glorot_uniform'))
        model.add(Activation('linear'))
        # loss
        model.compile(loss='mean_squared_error', optimizer="adam")
        # train
        model.fit(X_train[set_obj.index_base], set_obj.labels_train[set_obj.index_base] + 1,
                  nb_epoch=param['nb_epoch'], batch_size=param['batch_size'],
                  testation_split=0, verbose=0)
        # prediction
        pred = model.predict(X_test, verbose=0)
        pred.shape = (X_test.shape[0],)

        return pred

    @staticmethod
    def get_id():
        return "keras_dnn_model_id"

    @staticmethod
    def get_name():
        return "keras_dnn_model"
