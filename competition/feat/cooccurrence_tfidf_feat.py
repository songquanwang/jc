# coding:utf-8
"""
__file__

    genFeat_cooccurrence_tfidf.py

__description__

    This file generates the following features for each run and fold, and for the entire training and testing set.

        1. tfidf for the following cooccurrence terms
            - query unigram/bigram & title unigram/bigram
            - query unigram/bigram & description unigram/bigram
            - query id & title unigram/bigram
            - query id & description unigram/bigram

        2. corresponding lsa (svd) version features

__author__

    songquanwang

"""

import cPickle
import abc

from sklearn.decomposition import TruncatedSVD

from competition.feat.nlp.nlp_utils import getTFV
import competition.conf.model_params_conf as config
from  competition.feat.base_feat import BaseFeat


class CooccurenceTfidfFeat(BaseFeat):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        # cooccurrence terms column names
        self.column_names = [
            "query_unigram_title_unigram",
            "query_unigram_title_bigram",
            "query_unigram_description_unigram",
            "query_unigram_description_bigram",

            "query_bigram_title_unigram",
            "query_bigram_title_bigram",
            "query_bigram_description_unigram",
            "query_bigram_description_bigram",

            "query_id_title_unigram",
            "query_id_title_bigram",
            "query_id_description_unigram",
            "query_id_description_bigram",
        ]
        ## feature names
        self.feat_names = [name + "_tfidf" for name in self.column_names]
        ## file to save feat names
        self.feat_name_file = "%s/intersect_tfidf.feat_name" % config.feat_folder

        self.ngram_range = config.cooccurrence_tfidf_ngram_range

        self.svd_n_components = 100

    @staticmethod
    def cooccurrence_terms(lst1, lst2, join_str):
        """
            Cooccurrence terms：两组单词任意组合，用join_str链接
        :param lst1:
        :param lst2:
        :param join_str: len(lst1)*len(lst2) 长度的单词组合[],用空格join成一个字符串返回
        :return:
        """
        terms = [""] * len(lst1) * len(lst2)
        cnt = 0
        for item1 in lst1:
            for item2 in lst2:
                terms[cnt] = item1 + join_str + item2
                cnt += 1
        res = " ".join(terms)
        return res

    def extract_feat(self, df):
        cooccurrence_terms = CooccurenceTfidfFeat.cooccurrence_terms
        ## cooccurrence terms
        join_str = "X"
        # query_unigram * [titile,description] =4
        df["query_unigram_title_unigram"] = list(df.apply(lambda x: cooccurrence_terms(x["query_unigram"], x["title_unigram"], join_str), axis=1))
        df["query_unigram_title_bigram"] = list(df.apply(lambda x: cooccurrence_terms(x["query_unigram"], x["title_bigram"], join_str), axis=1))
        df["query_unigram_description_unigram"] = list(df.apply(lambda x: cooccurrence_terms(x["query_unigram"], x["description_unigram"], join_str), axis=1))
        df["query_unigram_description_bigram"] = list(df.apply(lambda x: cooccurrence_terms(x["query_unigram"], x["description_bigram"], join_str), axis=1))
        # query_bigram * [titile,description] =4
        df["query_bigram_title_unigram"] = list(df.apply(lambda x: cooccurrence_terms(x["query_bigram"], x["title_unigram"], join_str), axis=1))
        df["query_bigram_title_bigram"] = list(df.apply(lambda x: cooccurrence_terms(x["query_bigram"], x["title_bigram"], join_str), axis=1))
        df["query_bigram_description_unigram"] = list(df.apply(lambda x: cooccurrence_terms(x["query_bigram"], x["description_unigram"], join_str), axis=1))
        df["query_bigram_description_bigram"] = list(df.apply(lambda x: cooccurrence_terms(x["query_bigram"], x["description_bigram"], join_str), axis=1))
        # query_id * [titile,description] =4
        df["query_id_title_unigram"] = list(df.apply(lambda x: cooccurrence_terms(["qid" + str(x["qid"])], x["title_unigram"], join_str), axis=1))
        df["query_id_title_bigram"] = list(df.apply(lambda x: cooccurrence_terms(["qid" + str(x["qid"])], x["title_bigram"], join_str), axis=1))
        df["query_id_description_unigram"] = list(df.apply(lambda x: cooccurrence_terms(["qid" + str(x["qid"])], x["description_unigram"], join_str), axis=1))
        df["query_id_description_bigram"] = list(df.apply(lambda x: cooccurrence_terms(["qid" + str(x["qid"])], x["description_bigram"], join_str), axis=1))

    def gen_tfidf_svd_by_feat_column_names(self, path, dfTrain, dfTest, mode, ngram_range, feat_names, column_names):
        """
        只提取feat_names这些特征
        :param dfTrain:
        :param dfTest:
        :param mode:
        :param ngram_range:
        :param feat_names:
        :param column_names:
        :return:
        """
        for feat_name, column_name in zip(feat_names, column_names):
            print "generate %s feat" % feat_name
            ## tfidf
            tfv = getTFV(ngram_range=ngram_range)
            X_tfidf_train = tfv.fit_transform(dfTrain[column_name])
            X_tfidf_test = tfv.transform(dfTest[column_name])

            with open("%s/train.%s.feat.pkl" % (path, feat_name), "wb") as f:
                cPickle.dump(X_tfidf_train, f, -1)
            with open("%s/%s.%s.feat.pkl" % (path, mode, feat_name), "wb") as f:
                cPickle.dump(X_tfidf_test, f, -1)

            ## svd
            svd = TruncatedSVD(n_components=self.svd_n_components, n_iter=15)
            X_svd_train = svd.fit_transform(X_tfidf_train)
            X_svd_test = svd.transform(X_tfidf_test)
            with open("%s/train.%s_individual_svd%d.feat.pkl" % (path, feat_name, self.svd_n_components), "wb") as f:
                cPickle.dump(X_svd_train, f, -1)
            with open("%s/%s.%s_individual_svd%d.feat.pkl" % (path, mode, feat_name, self.svd_n_components), "wb") as f:
                cPickle.dump(X_svd_test, f, -1)

    def gen_coocurrence_tfidf_feat(self):

        # cooccurrence terms column names
        column_names = [
            "query_unigram_title_unigram",
            "query_unigram_title_bigram",
            "query_unigram_description_unigram",
            "query_unigram_description_bigram",

            "query_bigram_title_unigram",
            "query_bigram_title_bigram",
            "query_bigram_description_unigram",
            "query_bigram_description_bigram",

            "query_id_title_unigram",
            "query_id_title_bigram",
            "query_id_description_unigram",
            "query_id_description_bigram",
        ]
        ## feature names
        feat_names = [name + "_tfidf" for name in column_names]


        ngram_range = config.cooccurrence_tfidf_ngram_range

        svd_n_components = 100

        # Load Data
        with open(config.processed_train_data_path, "rb") as f:
            dfTrain = cPickle.load(f)
        with open(config.processed_test_data_path, "rb") as f:
            dfTest = cPickle.load(f)
        ## load pre-defined stratified k-fold index
        with open("%s/stratifiedKFold.%s.pkl" % (config.solution_data, config.stratified_label), "rb") as f:
            skf = cPickle.load(f)

        print("==================================================")
        print("Generate co-occurrence tfidf features...")

        # gen temp feat
        self.gen_temp_feat(dfTrain)
        self.gen_temp_feat(dfTest)
        # get cooccurrence terms
        self.extract_feat(dfTrain)
        self.extract_feat(dfTest)

        # Cross validation
        print("For cross-validation...")
        for run in range(config.n_runs):
            # use 33% for training and 67 % for validation ,so we switch trainInd and validInd
            for fold, (validInd, trainInd) in enumerate(skf[run]):
                print("Run: %d, Fold: %d" % (run + 1, fold + 1))
                path = "%s/Run%d/Fold%d" % (config.solution_feat_base, run + 1, fold + 1)
                X_tfidf_train = dfTrain.iloc[trainInd]
                X_tfidf_valid = dfTrain.iloc[validInd]
                self.gen_tfidf_svd_by_feat_column_names(path, X_tfidf_train, X_tfidf_valid, "valid", ngram_range, feat_names,
                                                        column_names)

        print("Done.")

        # Re-training
        print("For training and testing...")
        path = "%s/All" % config.solution_feat_base
        self.gen_tfidf_svd_by_feat_column_names(path, dfTrain, dfTest, "test", ngram_range, feat_names, column_names)
        feat_names += ["%s_individual_svd%d" % (f, svd_n_components) for f in feat_names]
        print("Done.")

        # 保存所有的特征名字：intersect_tfidf.feat_name
        feat_name_file = "%s/intersect_tfidf.feat_name" % config.solution_feat_combined
        print("Feature names are stored in %s" % feat_name_file)
        self.dump_feat_name(feat_names, feat_name_file)

        print("All Done.")
