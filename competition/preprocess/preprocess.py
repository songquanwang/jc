# coding:utf-8
"""
__file__

    preprocess.py

__description__

    This file preprocesses data.
    train 文件格式："id","query","product_title","product_description","median_relevance","relevance_variance"
    test  文件格式："id","query","product_title","product_description"

__author__

    songquanwang
    
"""

import os
import cPickle

import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedKFold

from competition.feat.nlp.nlp_utils import clean_text
import competition.conf.model_params_conf as config


def init_path():
    ## create feat folder
    if not os.path.exists(config.feat_folder):
        os.makedirs(config.feat_folder)

    ## creat folder for the training and testing feat
    if not os.path.exists("%s/All" % config.feat_folder):
        os.makedirs("%s/All" % config.feat_folder)

    ## creat folder for each run and fold
    for run in range(1, config.n_runs + 1):
        for fold in range(1, config.n_folds + 1):
            path = "%s/Run%d/Fold%d" % (config.feat_folder, run, fold)
            if not os.path.exists(path):
                os.makedirs(path)


def preprocess():
    """
    1.load  train and test data
   2.add index: 从0 开始编号
   3.dummy median_relevance_%d
   4.add qid  :query distinct 后的序号
   5.替换一些同义词，清除html标记
    """

    print("Load data...")

    dfTrain = pd.read_csv(config.original_train_data_path).fillna("")
    dfTest = pd.read_csv(config.original_test_data_path).fillna("")
    # number of train/test samples
    num_train, num_test = dfTrain.shape[0], dfTest.shape[0]

    print("Done.")


    ######################
    ## Pre-process Data ##
    ######################
    print("Pre-process data...")

    ## insert fake label for test
    dfTest["median_relevance"] = np.ones((num_test))
    dfTest["relevance_variance"] = np.zeros((num_test))

    ## insert sample index
    dfTrain["index"] = np.arange(num_train)
    dfTest["index"] = np.arange(num_test)

    ## one-hot encode the median_relevance ：dummy median_relevance
    for i in range(config.n_classes):
        dfTrain["median_relevance_%d" % (i + 1)] = 0
        dfTrain["median_relevance_%d" % (i + 1)][dfTrain["median_relevance"] == (i + 1)] = 1

    ## query ids
    qid_dict = dict()
    for i, q in enumerate(np.unique(dfTrain["query"]), start=1):
        qid_dict[q] = i

    ## insert query id
    dfTrain["qid"] = map(lambda q: qid_dict[q], dfTrain["query"])
    dfTest["qid"] = map(lambda q: qid_dict[q], dfTest["query"])

    ## clean text
    clean = lambda line: clean_text(line, drop_html_flag=config.drop_html_flag)
    dfTrain = dfTrain.apply(clean, axis=1)
    dfTest = dfTest.apply(clean, axis=1)

    print("Done.")


    ###############
    ## Save Data ##
    ###############
    print("Save data...")

    with open(config.processed_train_data_path, "wb") as f:
        cPickle.dump(dfTrain, f, -1)
    with open(config.processed_test_data_path, "wb") as f:
        cPickle.dump(dfTest, f, -1)

    print("Done.")

    """
    ## pos tag text
    dfTrain = dfTrain.apply(pos_tag_text, axis=1)
    dfTest = dfTest.apply(pos_tag_text, axis=1)
    with open(config.pos_tagged_train_data_path, "wb") as f:
        cPickle.dump(dfTrain, f, -1)
    with open(config.pos_tagged_test_data_path, "wb") as f:
        cPickle.dump(dfTest, f, -1)
    print("Done.")
    """


def gen_stratified_kfold():
    """
     This file generates the StratifiedKFold indices which will be kept fixed in
     ALL the following model building parts.
     分层抽取: 根据median_relevance 也就是 0 1 2 3 各种等级抽取近似；qid 不同的关键字抽取近似
     [
     [[validInd_fold1,trainInd_fold1],[validInd_fold1,trainInd_fold1],[validInd_fold1,trainInd_fold1]],
     [run2],
     [run3]
     ]
    """

    ## load data
    with open(config.processed_train_data_path, "rb") as f:
        dfTrain = cPickle.load(f)

    skf = [0] * config.n_runs
    for stratified_label, key in zip(["relevance", "query"], ["median_relevance", "qid"]):
        for run in range(config.n_runs):
            random_seed = 2015 + 1000 * (run + 1)
            skf[run] = StratifiedKFold(dfTrain[key], n_folds=config.n_folds,
                                       shuffle=True, random_state=random_seed)
            for fold, (validInd, trainInd) in enumerate(skf[run]):
                print("================================")
                print("Index for run: %s, fold: %s" % (run + 1, fold + 1))
                print("Train (num = %s)" % len(trainInd))
                print(trainInd[:10])
                print("Valid (num = %s)" % len(validInd))
                print(validInd[:10])
        with open("%s/stratifiedKFold.%s.pkl" % (config.data_folder, stratified_label), "wb") as f:
            cPickle.dump(skf, f, -1)
