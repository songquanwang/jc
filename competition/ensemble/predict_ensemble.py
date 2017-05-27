# coding:utf-8
"""
__file__

    ensemble_selection.py

__description__

    This file contains ensemble selection module.

__author__

    songquanwang

"""

import abc
import numpy as np
import pandas as pd
import os
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from competition.utils.utils import getScore, getTestScore
from competition.utils.ml_metrics import quadratic_weighted_kappa
import competition.conf.model_params_conf as config
import competition.conf.model_library_config as model_library_config
import competition.utils.utils as utils


class PredictEnsemble(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.model_folder = "../../Output"
        self.subm_folder = "../../Output/Subm"
        self.model_list = []
        self.model2idx = dict()
        # 每个model的kappa值
        self.kappa_list = dict()

        self.max_num_valid = 12000
        # 模型-run-fold-行
        self.pred_list_valid = np.zeros((len(self.model_list), config.n_runs, config.n_folds, self.max_num_valid), dtype=float)
        # run-fold-行
        self.y_list_valid = np.zeros((config.n_runs, config.n_folds, self.max_num_valid), dtype=float)
        # run-fold-4类别
        self.cdf_list_valid = np.zeros((config.n_runs, config.n_folds, config.n_classes), dtype=float)
        # run-fold
        self.num_valid_matrix = np.zeros((config.n_runs, config.n_folds), dtype=int)
        # run-fold-行
        self.p_ens_list_valid = np.zeros((config.n_runs, config.n_folds, self.max_num_valid), dtype=float)

        id_sizes = 10 * np.ones(len(model_library_config.feat_names), dtype=int)
        for feat_name, id_size in zip(model_library_config.feat_names, id_sizes):
            # 获取每个算法的前十个模型；根据kappa_mean排序
            log_file = "%s/Log/%s_hyperopt.log" % (self.model_folder, feat_name)
            try:
                # 读取模型平均参数
                dfLog = pd.read_csv(log_file)
                # kappa mean降序排列
                dfLog.sort("kappa_mean", ascending=False, inplace=True)
                ind = np.min([id_size, dfLog.shape[0]])
                # 取出前十行 也就是当前模型的前十名尝试累加到数组
                ids = dfLog.iloc[:ind]["trial_counter"]
                # print dfLog[:ind]
                self.model_list += ["%s_[Id@%d]" % (feat_name, id) for id in ids]
            except:
                pass
        # 初始化
        for i, model in enumerate(self.model_list):
            self.model2idx[model] = i
            self.kappa_list[model] = 0

        # 不存在则创建目录
        if not os.path.exists(self.subm_folder):
            os.makedirs(self.subm_folder)

    def ensemble_selection_prediction(self, best_bagged_model_list, best_bagged_model_weight, cdf, cutoff=None):
        """
        按照bagging、model_list 集成预测结果
        :param best_bagged_model_list:
        :param best_bagged_model_weight:
        :param cdf:
        :param cutoff:
        :return:
        """
        bagging_size = len(best_bagged_model_list)
        for bagging_iter in range(bagging_size):
            # 初始化累计权重
            w_ens = 0
            iter = 0
            # 多个模型集成结果
            for model, w in zip(best_bagged_model_list[bagging_iter], best_bagged_model_weight[bagging_iter]):
                iter += 1
                pred_file = "%s/All/test.pred.%s.csv" % (self.model_folder, model)
                # 获取当前模型预测值
                this_p_valid = pd.read_csv(pred_file, dtype=float)["prediction"].values
                this_w = w
                if iter == 1:
                    # 初始化整合预测值是0
                    p_ens_valid = np.zeros((this_p_valid.shape[0]), dtype=float)
                    id_test = pd.read_csv(pred_file, dtype=float)["id"].values
                    id_test = np.asarray(id_test, dtype=int)
                # 按照归一化权重 线性组合
                p_ens_valid = (w_ens * p_ens_valid + this_w * this_p_valid) / (w_ens + this_w)
                # 累计权重
                w_ens += this_w
            # 多个bagging进行集成，每个bagging的权重都相同
            if bagging_iter == 0:
                p_ens_valid_bag = p_ens_valid
            else:
                # 每次bagging的权重都是1，同等权重
                p_ens_valid_bag = (bagging_iter * p_ens_valid_bag + p_ens_valid) / (bagging_iter + 1.)
        # 根据cdf对排序后的预测结果进行映射成1-4
        if cutoff is None:
            p_ens_score = getScore(p_ens_valid_bag, cdf)
        else:
            # 使用相近取整的方式得出预测结果
            p_ens_score = getTestScore(p_ens_valid_bag, cutoff)
        # 输出集成后的结果
        output = pd.DataFrame({"id": id_test, "prediction": p_ens_score})
        return output

    def gen_kappa_list(self, feat_folder, cdf):
        """

        :param feat_folder:
        :param cdf:
        :param numValidMatrix:  引用 可用临时变量替代，因为返回的是最后一个model的值
        :param pred_list_valid: 引用
        :param Y_list_valid: 引用   可用临时变量替代,因为返回的是最后一个model的值
        :param cdf_list_valid: 引用 可用临时变量替代, 因为返回的是最后一个model的值
        :param kappa_list: 引用
        :return:
        """
        print("Load model...")
        for i, model in enumerate(self.model_list):
            print("model: %s" % model)
            for run in range(config.n_runs):
                for fold in range(config.n_folds):
                    path = "%s/Run%d/Fold%d" % (self.model_folder, run + 1, fold + 1)
                    pred_file = "%s/valid.pred.%s.csv" % (path, model)
                    cdf_file = "%s/Run%d/Fold%d/valid.cdf" % (feat_folder, run + 1, fold + 1)
                    this_p_valid = pd.read_csv(pred_file, dtype=float)
                    # 这些指标只需要执行一次就行了，每个模型都一样
                    if i == 0:
                        # 记录run-fold的行数
                        self.num_valid_matrix[run][fold] = this_p_valid.shape[0]
                        # 记录run-fold的真实值
                        self.y_list_valid[run, fold, :self.num_valid_matrix[run][fold]] = this_p_valid["target"].values
                        # load cdf
                        if cdf == None:
                            self.cdf_list_valid[run, fold, :] = np.loadtxt(cdf_file, dtype=float)
                        else:
                            self.cdf_list_valid[run, fold, :] = cdf
                        score = getScore(this_p_valid["prediction"].values, self.cdf_list_valid[run, fold, :])
                        self.kappa_cv[run][fold] = quadratic_weighted_kappa(score, self.y_list_valid[run, fold, :self.num_valid_matrix[run][fold]])
                    # 记录model-run-fold的预测值数组
                    self.pred_list_valid[self.model2idx[model], run, fold, :this_p_valid.shape[0]] = this_p_valid["prediction"].values
            print("kappa: %.6f" % np.mean(self.kappa_cv))
            # 算出每个模型的平均kappa_cv
            self.kappa_list[model] = np.mean(self.kappa_cv)

    def gen_ens_topk(self, init_top_k, this_sorted_models):
        """
          选择前五个模型 返回整合后的预测值；前五个模型名字；前五个模型的权重(全是1)
        :param init_top_k:
        :param this_sorted_models:
        :param model2idx:
        :param pred_list_valid:
        :param numValidMatrix:
        :param cdf_list_valid:
        :param Y_list_valid:
        :return:
        """
        best_model_list = []
        best_model_weight = []
        p_ens_list_valid_topk = np.zeros((config.n_runs, config.n_folds, self.numValidMatrix), dtype=float)
        w_ens, this_w = 0, 1.0
        cnt = 0
        kappa_cv = np.zeros((config.n_runs, config.n_folds), dtype=float)
        for model, kappa in this_sorted_models[0:init_top_k]:
            print("add to the ensembles the following model")
            print("model: %s" % model)
            print("kappa: %.6f" % kappa)
            # 指定模型的预测结果
            this_p_list_valid = self.pred_list_valid[self.model2idx[model]]
            for run in range(config.n_runs):
                for fold in range(config.n_folds):
                    num_valid = self.num_valid_matrix[run][fold]
                    # 多个模型预测值线性组合
                    p_ens_list_valid_topk[run, fold, :num_valid] = (w_ens * p_ens_list_valid_topk[run, fold, :num_valid] + this_w * this_p_list_valid[run, fold, :num_valid]) / (w_ens + this_w)
                    # 在最后一个model，生成一些指标
                    if cnt == init_top_k - 1:
                        cdf = self.cdf_list_valid[run, fold, :]
                        true_label = self.Y_list_valid[run, fold, :num_valid]
                        score = getScore(p_ens_list_valid_topk[run, fold, :num_valid], cdf)
                        kappa_cv[run][fold] = quadratic_weighted_kappa(score, true_label)
            best_model_list.append(model)
            best_model_weight.append(this_w)
            w_ens += this_w
            cnt += 1
            print("Init kappa: %.6f (%.6f)" % (np.mean(kappa_cv), np.std(kappa_cv)))
        return best_model_list, best_model_weight, p_ens_list_valid_topk, w_ens

    def ensemble_selection_obj(self, param, p1_list, weight1, p2_list):

        """
        优化param中的weight_current_model参数，使其平均kappa_cv_mean
        :param param:
        :param p1_list: 集成前五个模型(也就是对前五个模型求平均值的结果)
        :param weight1: 1
        :param p2_list: 当前模型预测结果
        :return:
        """
        weight_current_model = param['weight_current_model']
        kappa_cv = np.zeros((config.n_runs, config.n_folds), dtype=float)
        for run in range(config.n_runs):
            for fold in range(config.n_folds):
                num_valid = self.num_valid_matrix[run][fold]
                p1 = p1_list[run, fold, :num_valid]
                p2 = p2_list[run, fold, :num_valid]
                true_label = self.y_list_valid[run, fold, :num_valid]
                cdf = self.cdf_list_valid[run, fold, :]
                p_ens = (weight1 * p1 + weight_current_model * p2) / (weight1 + weight_current_model)
                p_ens_score = getScore(p_ens, cdf)
                kappa_cv[run][fold] = quadratic_weighted_kappa(p_ens_score, true_label)
        kappa_cv_mean = np.mean(kappa_cv)
        return {'loss': -kappa_cv_mean, 'status': STATUS_OK}

    def find_best_model(self, this_sorted_models, w_ens, w_min, w_max, best_kappa, hypteropt_max_evals, p_ens_list_valid_topk):
        """
        寻找最佳模型、权重、kappa值
        :param this_sorted_models:
        :param w_ens:
        :param w_min:
        :param w_max:
        :param hypteropt_max_evals:
        :param p_ens_list_valid_topk:
        :return:
        """
        best_model = None
        best_weight = 0
        for model, kappa in this_sorted_models:
            # 当前模型预测值
            this_p_list_valid = self.pred_list_valid[self.model2idx[model]]
            # hyperopt 找当前模型最优权重
            trials = Trials()
            # 不同模型的权重
            param_space = {
                'weight_current_model': hp.uniform('weight_current_model', w_min, w_max)
            }
            obj = lambda param: self.ensemble_selection_obj(param, p_ens_list_valid_topk, 1., this_p_list_valid)
            best_params = fmin(obj, param_space, algo=tpe.suggest, trials=trials, max_evals=hypteropt_max_evals)
            # 返回当前模型权重
            this_w = best_params['weight_current_model']
            # 按比例缩放当前权重 1 this_w --- w_ens this_w * w_ens
            this_w *= w_ens
            # all the current prediction to the ensemble
            kappa_cv = np.zeros((config.n_runs, config.n_folds), dtype=float)
            for run in range(config.n_runs):
                for fold in range(config.n_folds):
                    num_valid = self.num_valid_matrix[run][fold]
                    # topk预测
                    p1 = p_ens_list_valid_topk[run, fold, :num_valid]
                    # 当前预测
                    p2 = this_p_list_valid[run, fold, :num_valid]
                    # 真实label
                    true_label = self.y_list_valid[run, fold, :num_valid]
                    cdf = self.cdf_list_valid[run, fold, :]
                    p_ens = (w_ens * p1 + this_w * p2) / (w_ens + this_w)
                    score = getScore(p_ens, cdf)
                    # 集成后kappa值
                    kappa_cv[run][fold] = quadratic_weighted_kappa(score, true_label)
            if np.mean(kappa_cv) > best_kappa:
                best_kappa, best_model, best_weight = np.mean(kappa_cv), model, this_w
        return best_kappa, best_model, best_weight

    def gen_best_weight(self, this_sorted_models, w_min, w_max, hypteropt_max_evals, w_ens, p_ens_list_valid_topk, best_model_list, best_model_weight):
        """
        一组模型，循环遍历，只集成比当前模型好的
        :param this_sorted_models:
        :param w_min:
        :param w_max:
        :param hypteropt_max_evals:
        :param w_ens: topk集成结果权重
        :param p_ens_list_valid_topk:topk集成结果
        :param best_model_list:引用
        :param best_model_weight:引用
        :return:
        """
        iter = 0
        best_kappa = 0
        best_model = None
        while True:
            iter += 1
            # 根据 当前best_kappa 找到一个比他平均kappa更好的模型;每次w_ens,best_kappa、p_ens_list_valid_topk 变化
            best_kappa, best_model, best_weight = self.find_best_model(this_sorted_models, w_ens, w_min, w_max, best_kappa, hypteropt_max_evals, p_ens_list_valid_topk)
            # 这次迭代找不到best_kappa更好的模型则终止
            if best_model == None:
                break
            print("Iter: %d" % iter)
            print("    model: %s" % best_model)
            print("    weight: %s" % best_weight)
            print("    kappa: %.6f" % best_kappa)
            best_model_list.append(best_model)
            best_model_weight.append(best_weight)
            # 最优模型
            this_p_list_valid = self.pred_list_valid[self.model2idx[best_model]]
            for run in range(config.n_runs):
                for fold in range(config.n_folds):
                    num_valid = self.num_valid_matrix[run][fold]
                    # 历史累计集成结果和当前（比历史上最佳好的模型）的集成结果
                    p_ens_list_valid_topk[run, fold, :num_valid] = (w_ens * p_ens_list_valid_topk[run, fold, :num_valid] + best_weight * this_p_list_valid[run, fold, :num_valid]) / (w_ens + best_weight)
            best_model = None
            w_ens += best_weight

    def gen_kappa_cv(self, bagging_iter, p_ens_list_valid, p_ens_list_valid_topk):
        """
        
        :param bagging_iter:
        :param p_ens_list_valid:
        :param p_ens_list_valid_topk:
        :return:
        """

        kappa_cv = np.zeros((config.n_runs, config.n_folds), dtype=float)
        cutoff = np.zeros((3), dtype=float)
        for run in range(config.n_runs):
            for fold in range(config.n_folds):
                numValid = self.num_valid_matrix[run][fold]
                true_label = self.y_list_valid[run, fold, :numValid]
                cdf = self.cdf_list_valid[run, fold, :]
                p_ens_list_valid[run, fold, :numValid] = (bagging_iter * p_ens_list_valid[run, fold, :numValid] + p_ens_list_valid_topk[run, fold, :numValid]) / (bagging_iter + 1.)
                score, cutoff_tmp = getScore(p_ens_list_valid[run, fold, :numValid], cdf, "valid")
                kappa_cv[run][fold] = quadratic_weighted_kappa(score, true_label)
                cutoff += cutoff_tmp
        cutoff /= float(config.n_runs * config.n_folds)
        cutoff *= (22513 / ((2. / 3) * 10158))
        print("Bag %d, kappa: %.6f (%.6f)" % (bagging_iter + 1, np.mean(kappa_cv), np.std(kappa_cv)))
        return kappa_cv, cutoff

    def ensemble_selection(self, feat_folder, model_folder, model_list, cdf, cdf_test, subm_prefix, hypteropt_max_evals=10, w_min=-1., w_max=1., bagging_replacement=False, bagging_fraction=0.5, bagging_size=10,
                           init_top_k=5,
                           prunning_fraction=0.2):
        """

        :param feat_folder:
        :param model_folder:
        :param model_list:
        :param cdf:
        :param cdf_test:
        :param subm_prefix:
        :param hypteropt_max_evals:
        :param w_min:
        :param w_max:
        :param bagging_replacement:
        :param bagging_fraction:
        :param bagging_size:
        :param init_top_k:
        :param prunning_fraction:
        :return:
        """

        print("Load model...")
        self.gen_kappa_list(feat_folder, cdf)

        cdf_mean_init = np.mean(np.mean(self.cdf_list_valid, axis=0), axis=0)
        cdf_mean_init = cdf_mean_init.tolist()
        cdf_mean_init.insert(0, 0)
        # diff  1 2 4 --->1 2  ;把累计值拆分成 各个区间值
        pdf_mean_init = np.diff(np.asarray(cdf_mean_init))
        # 按照 kappa_cv值 排序-置逆；大的靠前 ('model_id',cv)
        sorted_models = sorted(self.kappa_list.items(), key=lambda x: x[1])[::-1]

        # greedy ensemble
        print("============================================================")
        print("Perform ensemble selection...")
        best_bagged_model_list = [[]] * bagging_size
        best_bagged_model_weight = [[]] * bagging_size
        num_model = len(model_list)
        # print bagging_size
        for bagging_iter in range(bagging_size):
            index_base, index_meta = utils.bootstrap_data(2015 + 100 * bagging_iter, bagging_replacement, num_model, bagging_fraction)
            this_sorted_models = [sorted_models[i] for i in sorted(index_base)]

            # initialization
            best_model_list, best_model_weight, p_ens_list_valid_topk, w_ens = self.gen_ens_topk(init_top_k, this_sorted_models)

            #### ensemble selection with replacement
            self.gen_best_weight(this_sorted_models, w_min, w_max, hypteropt_max_evals, w_ens, p_ens_list_valid_topk, best_model_list, best_model_weight)

            kappa_cv, cutoff = self.gen_kappa_cv(self, bagging_iter, self.p_ens_list_valid, p_ens_list_valid_topk)

            best_kappa_mean = np.mean(kappa_cv)
            best_kappa_std = np.std(kappa_cv)
            best_bagged_model_list[bagging_iter] = best_model_list
            best_bagged_model_weight[bagging_iter] = best_model_weight

            ## save the current prediction
            output = self.ensemble_selection_prediction(model_folder, best_bagged_model_list[:(bagging_iter + 1)], best_bagged_model_weight[:(bagging_iter + 1)], cdf_test)
            sub_file = "%s_[InitTopK%d]_[BaggingSize%d]_[BaggingFraction%s]_[Mean%.6f]_[Std%.6f]_cdf.csv" % (subm_prefix, init_top_k, bagging_iter + 1, bagging_fraction, best_kappa_mean, best_kappa_std)
            output.to_csv(sub_file, index=False)
            # use cutoff
            output = self.ensemble_selection_prediction(model_folder, best_bagged_model_list[:(bagging_iter + 1)], best_bagged_model_weight[:(bagging_iter + 1)], cdf_test, cutoff)
            sub_file = "%s_[InitTopK%d]_[BaggingSize%d]_[BaggingFraction%s]_[Mean%.6f]_[Std%.6f]_cutoff.csv" % (subm_prefix, init_top_k, bagging_iter + 1, bagging_fraction, best_kappa_mean, best_kappa_std)
            output.to_csv(sub_file, index=False)
        return best_kappa_mean, best_kappa_std, best_bagged_model_list, best_bagged_model_weight

    def gen_ensemble(self,feat_folder):
        """

        :param feat_folder: '../../Feat/solution/LSA_and_stats_feat_Jun09'
        :return:
        """
        cdf_test = np.loadtxt("%s/All/test.cdf" % feat_folder, dtype=float)
        cdf_valid = None
        bagging_size = 100
        bagging_fraction = 1.0
        # 剪枝参数没有用到
        prunning_fraction = 1.
        bagging_replacement = True
        init_top_k = 5
        subm_prefix = "%s/test.pred.[ensemble_selection]_[Solution]" % (self.subm_folder)
        best_kappa_mean, best_kappa_std, best_bagged_model_list, best_bagged_model_weight = self.ensemble_selection(feat_folder, self.model_folder, self.model_list, cdf=cdf_valid, cdf_test=cdf_test, subm_prefix=subm_prefix, \
                                                                                                                    hypteropt_max_evals=1, w_min=-1, w_max=1, bagging_replacement=bagging_replacement,
                                                                                                                    bagging_fraction=bagging_fraction, \
                                                                                                                    bagging_size=bagging_size, init_top_k=init_top_k, prunning_fraction=prunning_fraction)
        return best_kappa_mean, best_kappa_std, best_bagged_model_list, best_bagged_model_weight
