# coding:utf-8
"""
__file__

    ensemble_selection.py

__description__

    This file contains ensemble selection module.

__author__

    Chenglong Chen < c.chenglong@gmail.com >

"""

import numpy as np
import pandas as pd
import os
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from competition.utils.utils import getScore, getTestScore
from competition.utils.ml_metrics import quadratic_weighted_kappa
import competition.conf.model_library_config as config


def ensembleSelectionPrediction(model_folder, best_bagged_model_list, best_bagged_model_weight, cdf, cutoff=None):
    """
    按照bagging、model_list 集成预测结果
    :param model_folder:
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
            pred_file = "%s/All/test.pred.%s.csv" % (model_folder, model)
            # 获取当前模型预测值
            this_p_valid = pd.read_csv(pred_file, dtype=float)["prediction"].values
            this_w = w
            if iter == 1:
                # 初始化整合预测值是0
                p_ens_valid = np.zeros((this_p_valid.shape[0]), dtype=float)
                id_test = pd.read_csv(pred_file, dtype=float)["id"].values
                id_test = np.asarray(id_test, dtype=int)
            # 按照权重比值相加，然后再归一化
            p_ens_valid = (w_ens * p_ens_valid + this_w * this_p_valid) / (w_ens + this_w)
            # 累计权重
            w_ens += this_w
        # 多个bagging进行集成，每个bagging的权重都相同
        if bagging_iter == 0:
            p_ens_valid_bag = p_ens_valid
        else:
            # 每次bagging的权重都是1；多次bagging后的整合结果权重累加
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


def gen_kappa_list(model_list, model2idx, model_folder, feat_folder, cdf, pred_list_valid, Y_list_valid, cdf_list_valid, numValidMatrix, kappa_list):
    """

    :param model_list:
    :param model2idx:
    :param model_folder:
    :param feat_folder:
    :param cdf:
    :param numValidMatrix:  引用
    :param pred_list_valid: 引用
    :param Y_list_valid: 引用
    :param cdf_list_valid: 引用
    :param kappa_list: 引用
    :return:
    """
    print("Load model...")
    for model in model_list:
        model_id = model2idx[model]
        print("model: %s" % model)
        kappa_cv = np.zeros((config.n_runs, config.n_folds), dtype=float)
        ## load cvf
        for run in range(config.n_runs):
            for fold in range(config.n_folds):
                path = "%s/Run%d/Fold%d" % (model_folder, run + 1, fold + 1)
                pred_file = "%s/valid.pred.%s.csv" % (path, model)
                cdf_file = "%s/Run%d/Fold%d/valid.cdf" % (feat_folder, run + 1, fold + 1)
                this_p_valid = pd.read_csv(pred_file, dtype=float)
                numValidMatrix[run][fold] = this_p_valid.shape[0]
                pred_list_valid[model_id, run, fold, :numValidMatrix[run][fold]] = this_p_valid["prediction"].values
                Y_list_valid[run, fold, :numValidMatrix[run][fold]] = this_p_valid["target"].values
                ## load cdf
                if cdf == None:
                    cdf_list_valid[run, fold, :] = np.loadtxt(cdf_file, dtype=float)
                else:
                    cdf_list_valid[run, fold, :] = cdf
                ##
                score = getScore(pred_list_valid[model_id, run, fold, :numValidMatrix[run][fold]], cdf_list_valid[run, fold, :])
                kappa_cv[run][fold] = quadratic_weighted_kappa(score, Y_list_valid[run, fold, :numValidMatrix[run][fold]])

        print("kappa: %.6f" % np.mean(kappa_cv))
        # 算出每个模型的平均kappa_cv
        kappa_list[model] = np.mean(kappa_cv)


def gen_ens_temp(init_top_k, this_sorted_models, model2idx, pred_list_valid, numValidMatrix, cdf_list_valid, Y_list_valid, p_ens_list_valid_tmp, best_model_list, best_model_weight):
    """

    :param init_top_k:
    :param this_sorted_models:
    :param model2idx:
    :param pred_list_valid:
    :param numValidMatrix:
    :param cdf_list_valid:
    :param Y_list_valid:
    :param p_ens_list_valid_tmp: 引用
    :param best_model_list: 引用
    :param best_model_weight: 引用
    :return:
    """
    #### initialization
    w_ens, this_w = 0, 1.0
    if init_top_k > 0:
        cnt = 0
        kappa_cv = np.zeros((config.n_runs, config.n_folds), dtype=float)
        for model, kappa in this_sorted_models:
            if cnt >= init_top_k:
                continue
            print("add to the ensembles the following model")
            print("model: %s" % model)
            print("kappa: %.6f" % kappa)
            this_p_list_valid = pred_list_valid[model2idx[model]]
            for run in range(config.n_runs):
                for fold in range(config.n_folds):
                    numValid = numValidMatrix[run][fold]
                    if cnt == 0:
                        this_w = 1.0
                    else:
                        pass
                    p_ens_list_valid_tmp[run, fold, :numValid] = (w_ens * p_ens_list_valid_tmp[run, fold, :numValid] + this_w * this_p_list_valid[run, fold, :numValid]) / (w_ens + this_w)
                    # p_ens_list_valid_tmp[run,fold,:numValid] = p_ens_list_valid_tmp[run,fold,:numValid].argsort().argsort()
                    if cnt == init_top_k - 1:
                        cdf = cdf_list_valid[run, fold, :]
                        true_label = Y_list_valid[run, fold, :numValid]
                        score = getScore(p_ens_list_valid_tmp[run, fold, :numValid], cdf)
                        kappa_cv[run][fold] = quadratic_weighted_kappa(score, true_label)
            best_model_list.append(model)
            best_model_weight.append(this_w)
            w_ens += this_w
            cnt += 1
        print("Init kappa: %.6f (%.6f)" % (np.mean(kappa_cv), np.std(kappa_cv)))
    return w_ens


def ensembleSelectionObj(param, p1_list, weight1, p2_list, true_label_list, cdf_list, numValidMatrix):
    """
    优化param中的weight2参数，使其平均kappa_cv_mean
    :param param:
    :param p1_list:
    :param weight1:
    :param p2_list:
    :param true_label_list:
    :param cdf_list:
    :param numValidMatrix:
    :return:
    """
    weight2 = param['weight2']
    kappa_cv = np.zeros((config.n_runs, config.n_folds), dtype=float)
    for run in range(config.n_runs):
        for fold in range(config.n_folds):
            numValid = numValidMatrix[run][fold]
            p1 = p1_list[run, fold, :numValid]
            p2 = p2_list[run, fold, :numValid]
            true_label = true_label_list[run, fold, :numValid]
            cdf = cdf_list[run, fold, :]
            p_ens = (weight1 * p1 + weight2 * p2) / (weight1 + weight2)
            p_ens_score = getScore(p_ens, cdf)
            kappa_cv[run][fold] = quadratic_weighted_kappa(p_ens_score, true_label)
    kappa_cv_mean = np.mean(kappa_cv)
    return {'loss': -kappa_cv_mean, 'status': STATUS_OK}


def gen_best_weight(this_sorted_models, model2idx, w_min, w_max, pred_list_valid, hypteropt_max_evals, w_ens, Y_list_valid, cdf_list_valid, numValidMatrix, p_ens_list_valid_tmp, best_model_list, best_model_weight):
    """

    :param this_sorted_models:
    :param model2idx:
    :param w_min:
    :param w_max:
    :param pred_list_valid:
    :param hypteropt_max_evals:
    :param w_ens:
    :param Y_list_valid:
    :param cdf_list_valid:
    :param numValidMatrix:
    :param p_ens_list_valid_tmp:
    :param best_model_list: 引用
    :param best_model_weight: 引用
    :return:
    """
    iter = 0
    while True:
        iter += 1
        for model, _ in this_sorted_models:
            this_p_list_valid = pred_list_valid[model2idx[model]]

            ## hyperopt for the best weight
            trials = Trials()
            # 不同模型的权重
            param_space = {
                'weight2': hp.uniform('weight2', w_min, w_max)
            }
            obj = lambda param: ensembleSelectionObj(param, p_ens_list_valid_tmp, 1., this_p_list_valid, Y_list_valid, cdf_list_valid, numValidMatrix)
            best_params = fmin(obj, param_space, algo=tpe.suggest, trials=trials, max_evals=hypteropt_max_evals)
            this_w = best_params['weight2']
            this_w *= w_ens
            # all the current prediction to the ensemble
            kappa_cv = np.zeros((config.n_runs, config.n_folds), dtype=float)
            for run in range(config.n_runs):
                for fold in range(config.n_folds):
                    numValid = numValidMatrix[run][fold]
                    p1 = p_ens_list_valid_tmp[run, fold, :numValid]
                    p2 = this_p_list_valid[run, fold, :numValid]
                    true_label = Y_list_valid[run, fold, :numValid]
                    cdf = cdf_list_valid[run, fold, :]
                    p_ens = (w_ens * p1 + this_w * p2) / (w_ens + this_w)
                    score = getScore(p_ens, cdf)
                    kappa_cv[run][fold] = quadratic_weighted_kappa(score, true_label)
            if np.mean(kappa_cv) > best_kappa:
                best_kappa, best_model, best_weight = np.mean(kappa_cv), model, this_w
        if best_model == None:
            break
        print("Iter: %d" % iter)
        print("    model: %s" % best_model)
        print("    weight: %s" % best_weight)
        print("    kappa: %.6f" % best_kappa)

        best_model_list.append(best_model)
        best_model_weight.append(best_weight)
        # valid
        this_p_list_valid = pred_list_valid[model2idx[best_model]]
        for run in range(config.n_runs):
            for fold in range(config.n_folds):
                numValid = numValidMatrix[run][fold]
                p_ens_list_valid_tmp[run, fold, :numValid] = (w_ens * p_ens_list_valid_tmp[run, fold, :numValid] + best_weight * this_p_list_valid[run, fold, :numValid]) / (w_ens + best_weight)
        best_model = None
        w_ens += best_weight


def gen_kappa_cv(bagging_iter, Y_list_valid, cdf_list_valid, numValidMatrix, p_ens_list_valid, p_ens_list_valid_tmp):
    """

    :param bagging_iter:
    :param Y_list_valid:
    :param cdf_list_valid:
    :param numValidMatrix:
    :param p_ens_list_valid:
    :param p_ens_list_valid_tmp:
    :return:
    """
    kappa_cv = np.zeros((config.n_runs, config.n_folds), dtype=float)
    cutoff = np.zeros((3), dtype=float)
    for run in range(config.n_runs):
        for fold in range(config.n_folds):
            numValid = numValidMatrix[run][fold]
            true_label = Y_list_valid[run, fold, :numValid]
            cdf = cdf_list_valid[run, fold, :]
            p_ens_list_valid[run, fold, :numValid] = (bagging_iter * p_ens_list_valid[run, fold,
                                                                     :numValid] + p_ens_list_valid_tmp[run, fold,
                                                                                  :numValid]) / (bagging_iter + 1.)
            score, cutoff_tmp = getScore(p_ens_list_valid[run, fold, :numValid], cdf, "valid")
            kappa_cv[run][fold] = quadratic_weighted_kappa(score, true_label)

            cutoff += cutoff_tmp
    cutoff /= float(config.n_runs * config.n_folds)
    cutoff *= (22513 / ((2. / 3) * 10158))
    print("Bag %d, kappa: %.6f (%.6f)" % (bagging_iter + 1, np.mean(kappa_cv), np.std(kappa_cv)))
    return kappa_cv, cutoff


def ensembleSelection(feat_folder, model_folder, model_list, cdf, cdf_test, subm_prefix, hypteropt_max_evals=10, w_min=-1., w_max=1., bagging_replacement=False, bagging_fraction=0.5, bagging_size=10, init_top_k=5,
                      prunning_fraction=0.2):
    ## load all the prediction :maxNumValid 预测结果假定是12000行
    maxNumValid = 12000
    # 模型-run-fold-行
    pred_list_valid = np.zeros((len(model_list), config.n_runs, config.n_folds, maxNumValid), dtype=float)
    # run-fold-行
    Y_list_valid = np.zeros((config.n_runs, config.n_folds, maxNumValid), dtype=float)
    # run-fold-4类别
    cdf_list_valid = np.zeros((config.n_runs, config.n_folds, config.n_classes), dtype=float)
    # run-fold
    numValidMatrix = np.zeros((config.n_runs, config.n_folds), dtype=int)
    # run-fold-行
    p_ens_list_valid = np.zeros((config.n_runs, config.n_folds, maxNumValid), dtype=float)

    numTest = 22513
    # model 从0开始编号
    model2idx = dict()
    # 每个model的kappa值
    kappa_list = dict()
    for i, model in enumerate(model_list):
        model2idx[model] = i
        kappa_list[model] = 0
    print("============================================================")
    print("Load model...")
    gen_kappa_list(model_list, model2idx, model_folder, feat_folder, cdf, pred_list_valid, Y_list_valid, cdf_list_valid, numValidMatrix, kappa_list)

    cdf_mean_init = np.mean(np.mean(cdf_list_valid, axis=0), axis=0)
    cdf_mean_init = cdf_mean_init.tolist()
    cdf_mean_init.insert(0, 0)
    # diff  1 2 4 --->1 2  ;把累计值拆分成 各个区间值
    pdf_mean_init = np.diff(np.asarray(cdf_mean_init))
    # 按照  kappa_cv值 排序-置逆；大的靠前
    sorted_models = sorted(kappa_list.items(), key=lambda x: x[1])[::-1]

    # greedy ensemble
    print("============================================================")
    print("Perform ensemble selection...")
    best_bagged_model_list = [[]] * bagging_size
    best_bagged_model_weight = [[]] * bagging_size
    num_model = len(model_list)
    # print bagging_size
    for bagging_iter in range(bagging_size):
        rng = np.random.RandomState(2015 + 100 * bagging_iter)
        if bagging_replacement:
            # 随机抽取
            sampleSize = int(num_model * bagging_fraction)
            index_base = rng.randint(num_model, size=sampleSize)
        else:
            # 均匀分布
            randnum = rng.uniform(size=num_model)
            index_base = [i for i in range(num_model) if randnum[i] < bagging_fraction]
        this_sorted_models = [sorted_models[i] for i in sorted(index_base)]

        # print this_model_list
        best_model_list = []
        best_model_weight = []
        best_kappa = 0
        best_model = None
        p_ens_list_valid_tmp = np.zeros((config.n_runs, config.n_folds, maxNumValid), dtype=float)
        # initialization
        w_ens = gen_ens_temp(init_top_k, this_sorted_models, model2idx, pred_list_valid, numValidMatrix, cdf_list_valid, Y_list_valid, p_ens_list_valid_tmp, best_model_list, best_model_weight)

        #### ensemble selection with replacement
        gen_best_weight(this_sorted_models, model2idx, w_min, w_max, pred_list_valid, hypteropt_max_evals, w_ens, Y_list_valid, cdf_list_valid, numValidMatrix, p_ens_list_valid_tmp, best_model_list, best_model_weight)

        kappa_cv, cutoff = gen_kappa_cv(bagging_iter, Y_list_valid, cdf_list_valid, numValidMatrix, p_ens_list_valid, p_ens_list_valid_tmp)

        best_kappa_mean = np.mean(kappa_cv)
        best_kappa_std = np.std(kappa_cv)
        best_bagged_model_list[bagging_iter] = best_model_list
        best_bagged_model_weight[bagging_iter] = best_model_weight

        ## save the current prediction
        # use cdf
        output = ensembleSelectionPrediction(model_folder, best_bagged_model_list[:(bagging_iter + 1)], best_bagged_model_weight[:(bagging_iter + 1)], cdf_test)
        sub_file = "%s_[InitTopK%d]_[BaggingSize%d]_[BaggingFraction%s]_[Mean%.6f]_[Std%.6f]_cdf.csv" % (subm_prefix, init_top_k, bagging_iter + 1, bagging_fraction, best_kappa_mean, best_kappa_std)
        output.to_csv(sub_file, index=False)
        # use cutoff
        output = ensembleSelectionPrediction(model_folder, best_bagged_model_list[:(bagging_iter + 1)], best_bagged_model_weight[:(bagging_iter + 1)], cdf_test, cutoff)
        sub_file = "%s_[InitTopK%d]_[BaggingSize%d]_[BaggingFraction%s]_[Mean%.6f]_[Std%.6f]_cutoff.csv" % (subm_prefix, init_top_k, bagging_iter + 1, bagging_fraction, best_kappa_mean, best_kappa_std)
        output.to_csv(sub_file, index=False)
    return best_kappa_mean, best_kappa_std, best_bagged_model_list, best_bagged_model_weight

if __name__ == "__main__":
    ##
    ## config
    model_folder = "../../Output"
    subm_folder = "../../Output/Subm"
    if not os.path.exists(subm_folder):
        os.makedirs(subm_folder)


    ## load test info
    feat_folder = config.feat_folders[0]
    info_test = pd.read_csv("%s/All/test.info" % feat_folder)
    id_test = info_test["id"]
    numTest = info_test.shape[0]
    ## load cdf
    cdf_test = np.loadtxt("%s/All/test.cdf" % feat_folder, dtype=float)

    # pdf_test = [0.089, 0.112, 0.19, 0.609]
    # cdf_test = np.cumsum(pdf_test)
    # cdf_valid = cdf_test
    cdf_valid = None


    ## reg
    model_list = []
    # try 5/10/50
    id_sizes = 10 * np.ones(len(config.feat_names), dtype=int)
    for feat_name, id_size in zip(config.feat_names, id_sizes):
        ## get the top 10 model ids
        log_file = "%s/Log/%s_hyperopt.log" % (model_folder, feat_name)
        try:
            # 读取模型平均参数
            dfLog = pd.read_csv(log_file)
            # kappa mean降序排列
            dfLog.sort("kappa_mean", ascending=False, inplace=True)
            ind = np.min([id_size, dfLog.shape[0]])
            # 取出前十行 也就是当前模型的前十名尝试累加到数组
            ids = dfLog.iloc[:ind]["trial_counter"]
            # print dfLog[:ind]
            model_list += ["%s_[Id@%d]" % (feat_name, id) for id in ids]
        except:
            pass

    bagging_size = 100
    bagging_fraction = 1.0
    prunning_fraction = 1.
    bagging_replacement = True
    init_top_k = 5

    subm_prefix = "%s/test.pred.[ensemble_selection]_[Solution]" % (subm_folder)
    best_kappa_mean, best_kappa_std, best_bagged_model_list, best_bagged_model_weight = ensembleSelection(feat_folder, model_folder, model_list, cdf=cdf_valid, cdf_test=cdf_test, subm_prefix=subm_prefix, \
                          hypteropt_max_evals=1, w_min=-1, w_max=1, bagging_replacement=bagging_replacement,
                          bagging_fraction=bagging_fraction, \
                          bagging_size=bagging_size, init_top_k=init_top_k, prunning_fraction=prunning_fraction)