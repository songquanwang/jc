# coding:utf-8
"""
__file__

    genFeat_basic_tfidf_feat.py

__description__

    This file generates the following features for each run and fold, and for the entire training and testing set.

        1. basic tfidf features for query/title/description
            - use common vocabulary among query/title/description for further computation of cosine similarity

        2. cosine similarity between query & title, query & description, title & description pairs
            - just plain cosine similarity

        3. cosine similarity stats features for title/description
            - computation is carried out with regard to a pool of samples grouped by:
                - median_relevance (#4)
                - query (qid) & median_relevance (#4)
            - cosine similarity for the following pairs are computed for each sample
                - sample title        vs.  pooled sample titles
                - sample description  vs.  pooled sample descriptions
                Note that in the pool samples, we exclude the current sample being considered.
            - stats features include quantiles of cosine similarity and others defined in the variable "stats_func", e.g.,
                - mean value
                - standard deviation (std)
                - more can be added, e.g., moment features etc

        4. SVD version of the above features

__author__

    Chenglong Chen < c.chenglong@gmail.com >

"""

import cPickle
from copy import copy

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances

from scipy.sparse import vstack
from competition.feat.nlp.nlp_utils import getTFV, getBOW

import competition.conf.model_params_conf as config
import abc
from  competition.feat.base_feat import BaseFeat


class BasicTfidfFeat(BaseFeat):
    __metaclass__ = abc.ABCMeta

    def __init__(self, stats_feat_flag=True):
        # 分位数，分别计算 0 0.5 1 分位数。 也就是 最小值、中位数、最大值
        quantiles_range = np.arange(0, 1.5, 0.5)
        # 是否计算统计特征
        self.stats_feat_flag = stats_feat_flag
        # 平均值 、标准差
        self.stats_func = [np.mean, np.std]
        # 特征包括 最小值 中位数 最大值，平均值、标准差 五个
        self.stats_feat_num = len(quantiles_range) + len(self.stats_func)
        # tfidf bow两种
        self.vec_types = ["tfidf", "bow"]
        # 1-3
        self.ngram_range = config.basic_tfidf_ngram_range
        # 默认是"common"，有两种选择'common','dinididual'
        self.vocabulary_type = config.basic_tfidf_vocabulary_type
        # 主成分个数
        self.svd_n_components = [100, 150]
        # 降维 ，没用到
        self.tsne_n_components = [2]

        # 三个列名
        self.column_names = ["query", "product_title", "product_description"]

    @staticmethod
    def cosine_sim(x, y):
        """
        计算余弦相似性距离
        :param x:
        :param y:
        :return:
        """
        try:
            d = cosine_similarity(x, y)
            d = d[0][0]
        except:
            print x
            print y
            d = 0.
        return d

    ## generate distance stats feat
    def generate_dist_stats_feat(self, metric, X_train, ids_train, X_test, ids_test, indices_dict, qids_test=None):
        """
        生成距离状态特征，每一行是一个向量，计算行之间的距离
        相互距离的最小值，中位数、最大值、平均值、方差
        :param metric: 距离度量标准 cosine/euclidean
        :param X_train:
        :param ids_train:
        :param X_test:
        :param ids_test:
        :param indices_dict:类别键值字典
        :param qids_test: 类别+qid键值字典
        :return:  len(ids_test) 行 stats_feat_num*n_classes列的矩阵
        stats_func ：全局函数
        stats_feat_num：全局

        """

        if metric == "cosine":
            # 生成 len(ids_test)行，class(分类个数)列的 多维数组
            stats_feat = 0 * np.ones((len(ids_test), self.stats_feat_num * config.n_classes), dtype=float)
            # sim 0-1 1完全相同
            sim = 1. - pairwise_distances(X_test, X_train, metric=metric, n_jobs=1)
        elif metric == "euclidean":
            stats_feat = -1 * np.ones((len(ids_test), self.stats_feat_num * config.n_classes), dtype=float)
            # 返回xtest行 xtrain列的array
            sim = pairwise_distances(X_test, X_train, metric=metric, n_jobs=1)

        for i in range(len(ids_test)):
            id = ids_test[i]
            if qids_test is not None:
                qid = qids_test[i]
            # 一行分别于某一类的距离做比较
            for j in range(config.n_classes):
                # if赋值语句
                key = (qid, j + 1) if qids_test is not None else j + 1
                if indices_dict.has_key(key):
                    inds = indices_dict[key]
                    # exclude this sample itself from the list of indices

                    inds = [ind for ind in inds if id != ids_train[ind]]
                    sim_tmp = sim[i][inds]
                    if len(sim_tmp) != 0:
                        # 距离的平均值、方差
                        feat = [func(sim_tmp) for func in self.stats_func]
                        ## quantile
                        sim_tmp = pd.Series(sim_tmp)
                        # 距离的最小值、中位数、最大值
                        quantiles = sim_tmp.quantile(self.quantiles_range)
                        feat = np.hstack((feat, quantiles))
                        # 每一行生成 [五个值的数组]
                        stats_feat[i, j * self.stats_feat_num:(j + 1) * self.stats_feat_num] = feat
        return stats_feat

    def extract_bow_tfidf_cosine_sim_stats_feat(self, path, dfTrain, dfTest, feat_name, column_name, X_train, X_test, vec_type, mode, relevance_indices_dict, query_relevance_indices_dict):
        """
        bow/tfidf cosine sim stats feat
        :param path:
        :param feat_name:
        :param column_name:
        :param X_train:
        :param X_test:
        :param mode:
        :param relevance_indices_dict:
        :param query_relevance_indices_dict:
        :param new_feat_names:
        :return:
        vec_type:["tfidf", "bow"]
        """
        # 方法生成的新特征名字
        new_feat_names = []
        if column_name in ["product_title", "product_description"]:
            print "generate %s stats feat for %s" % (vec_type, column_name)
            ## train
            cosine_sim_stats_feat_by_relevance_train = self.generate_dist_stats_feat("cosine", X_train, dfTrain["id"].values, X_train, dfTrain["id"].values, relevance_indices_dict)
            cosine_sim_stats_feat_by_query_relevance_train = self.generate_dist_stats_feat("cosine", X_train, dfTrain["id"].values, X_train, dfTrain["id"].values, query_relevance_indices_dict, dfTrain["qid"].values)
            with open("%s/train.%s_cosine_sim_stats_feat_by_relevance.feat.pkl" % (path, feat_name), "wb") as f:
                cPickle.dump(cosine_sim_stats_feat_by_relevance_train, f, -1)
            with open("%s/train.%s_cosine_sim_stats_feat_by_query_relevance.feat.pkl" % (path, feat_name), "wb") as f:
                cPickle.dump(cosine_sim_stats_feat_by_query_relevance_train, f, -1)
            ## test
            cosine_sim_stats_feat_by_relevance_test = self.generate_dist_stats_feat("cosine", X_train, dfTrain["id"].values, X_test, dfTest["id"].values, relevance_indices_dict)
            cosine_sim_stats_feat_by_query_relevance_test = self.generate_dist_stats_feat("cosine", X_train, dfTrain["id"].values, X_test, dfTest["id"].values, query_relevance_indices_dict, dfTest["qid"].values)
            with open("%s/%s.%s_cosine_sim_stats_feat_by_relevance.feat.pkl" % (path, mode, feat_name), "wb") as f:
                cPickle.dump(cosine_sim_stats_feat_by_relevance_test, f, -1)
            with open("%s/%s.%s_cosine_sim_stats_feat_by_query_relevance.feat.pkl" % (path, mode, feat_name),
                      "wb") as f:
                cPickle.dump(cosine_sim_stats_feat_by_query_relevance_test, f, -1)

            new_feat_names.append("%s_cosine_sim_stats_feat_by_relevance" % feat_name)
            new_feat_names.append("%s_cosine_sim_stats_feat_by_query_relevance" % feat_name)
            return new_feat_names

    def extract_cosine_sim_feat(path, feat_names, mode, vec_type):
        """
        cosine sim feat
        :param path:
        :param feat_names:
            feat_names = ["query", "title", "description"]
            vocabulary_type ="common"
            vec_type =["tfidf", "bow"]
            feat_names = [name + "_%s_%s_vocabulary" % (vec_type, vocabulary_type) for name in feat_names]
        :param mode:"valid"、"test"
        :param new_feat_names:
        :return:
        """
        # 方法生成的新特征名字
        new_feat_names = []
        for i in range(len(feat_names) - 1):
            for j in range(i + 1, len(feat_names)):
                print "generate common %s cosine sim feat for %s and %s" % (vec_type, feat_names[i], feat_names[j])
                for mod in ["train", mode]:
                    with open("%s/%s.%s.feat.pkl" % (path, mod, feat_names[i]), "rb") as f:
                        target_vec = cPickle.load(f)
                    with open("%s/%s.%s.feat.pkl" % (path, mod, feat_names[j]), "rb") as f:
                        obs_vec = cPickle.load(f)
                    sim = np.asarray(map(BasicTfidfModel.cosine_sim, target_vec, obs_vec))[:, np.newaxis]
                    # 计算两个特征之间的余弦相似度
                    with open("%s/%s.%s_%s_%s_cosine_sim.feat.pkl" % (path, mod, feat_names[i], feat_names[j], vec_type),
                              "wb") as f:
                        cPickle.dump(sim, f, -1)
                ## update feat names
                new_feat_names.append("%s_%s_%s_cosine_sim" % (feat_names[i], feat_names[j], vec_type))

        return new_feat_names

    def extract_svd_cosine_sim_stats_feat(self, path, dfTrain, dfTest, feat_name, column_name, X_svd_train, X_svd_test, mode, n_components, relevance_indices_dict, query_relevance_indices_dict):
        """
        svd sim stats feat
        :param path:
        :param feat_name:
        :param column_name:
        :param X_svd_train:
        :param X_svd_test:
        :param mode:
        :param n_components:
        :param relevance_indices_dict:
        :param query_relevance_indices_dict:
        :param new_feat_names:
        :return:
        """
        # 方法生成的新特征名字
        new_feat_names = []
        if column_name in ["product_title", "product_description"]:
            print "generate common %s-svd%d stats feat for %s" % (self.vec_type, n_components, column_name)
            ## train
            cosine_sim_stats_feat_by_relevance_train = self.generate_dist_stats_feat("cosine", X_svd_train, dfTrain["id"].values, X_svd_train, dfTrain["id"].values, relevance_indices_dict)
            cosine_sim_stats_feat_by_query_relevance_train = self.generate_dist_stats_feat("cosine", X_svd_train, dfTrain["id"].values, X_svd_train, dfTrain["id"].values, query_relevance_indices_dict,
                                                                                           dfTrain["qid"].values)
            with open("%s/train.%s_common_svd%d_cosine_sim_stats_feat_by_relevance.feat.pkl" % (path, feat_name, n_components), "wb") as f:
                cPickle.dump(cosine_sim_stats_feat_by_relevance_train, f, -1)
            with open("%s/train.%s_common_svd%d_cosine_sim_stats_feat_by_query_relevance.feat.pkl" % (path, feat_name, n_components), "wb") as f:
                cPickle.dump(cosine_sim_stats_feat_by_query_relevance_train, f, -1)
            ## test
            cosine_sim_stats_feat_by_relevance_test = self.generate_dist_stats_feat("cosine", X_svd_train, dfTrain["id"].values, X_svd_test, dfTest["id"].values, relevance_indices_dict)
            cosine_sim_stats_feat_by_query_relevance_test = self.generate_dist_stats_feat("cosine", X_svd_train, dfTrain["id"].values, X_svd_test, dfTest["id"].values, query_relevance_indices_dict, dfTest["qid"].values)
            with open("%s/%s.%s_common_svd%d_cosine_sim_stats_feat_by_relevance.feat.pkl" % (path, mode, feat_name, n_components), "wb") as f:
                cPickle.dump(cosine_sim_stats_feat_by_relevance_test, f, -1)
            with open("%s/%s.%s_common_svd%d_cosine_sim_stats_feat_by_query_relevance.feat.pkl" % (path, mode, feat_name, n_components), "wb") as f:
                cPickle.dump(cosine_sim_stats_feat_by_query_relevance_test, f, -1)
            ## update feat names
            new_feat_names.append("%s_common_svd%d_cosine_sim_stats_feat_by_relevance" % (feat_name, n_components))
            new_feat_names.append("%s_common_svd%d_cosine_sim_stats_feat_by_query_relevance" % (feat_name, n_components))

        return new_feat_names

    def extract_svd_cosine_sim_feat(path, feat_names, vec_type, mode, n_components):
        """
        svd cosine sim feat
        :param path:
        :param feat_names:
        :param mode:
        :param n_components:
        :param new_feat_names:
        :return:
        """
        # 方法生成的新特征名字
        new_feat_names = []
        for i in range(len(feat_names) - 1):
            for j in range(i + 1, len(feat_names)):
                print "generate common %s-svd%d cosine sim feat for %s and %s" % (vec_type, n_components, feat_names[i], feat_names[j])
            for mod in ["train", mode]:
                with open("%s/%s.%s_common_svd%d.feat.pkl" % (path, mod, feat_names[i], n_components), "rb") as f:
                    target_vec = cPickle.load(f)
                with open("%s/%s.%s_common_svd%d.feat.pkl" % (path, mod, feat_names[j], n_components), "rb") as f:
                    obs_vec = cPickle.load(f)
                sim = np.asarray(map(BasicTfidfModel.cosine_sim, target_vec, obs_vec))[:, np.newaxis]
                ## dump feat
                with open("%s/%s.%s_%s_%s_common_svd%d_cosine_sim.feat.pkl" % (path, mod, feat_names[i], feat_names[j], vec_type, n_components), "wb") as f:
                    cPickle.dump(sim, f, -1)
            ## update feat names
            new_feat_names.append("%s_%s_%s_common_svd%d_cosine_sim" % (feat_names[i], feat_names[j], vec_type, n_components))

    def extract_svd_cosine_sim_stats_feat_individual(self, path, dfTrain, dfTest, feat_name, column_name, X_svd_train, X_svd_test, vec_type, mode, n_components, relevance_indices_dict, query_relevance_indices_dict):
        """
        好像跟extract_svd_cosine_sim_stats_feat 完全一样
        :param path:
        :param feat_name:
        :param column_name:
        :param X_svd_train:
        :param X_svd_test:
        :param mode:
        :param n_components:
        :param relevance_indices_dict:
        :param query_relevance_indices_dict:
        :param new_feat_names:
        :return:
        """
        # 方法生成的新特征名字
        new_feat_names = []
        if column_name in ["product_title", "product_description"]:
            print "generate individual %s-svd%d stats feat for %s" % (vec_type, n_components, column_name)
            ## train
            cosine_sim_stats_feat_by_relevance_train = self.generate_dist_stats_feat("cosine", X_svd_train, dfTrain["id"].values, X_svd_train, dfTrain["id"].values, relevance_indices_dict)
            cosine_sim_stats_feat_by_query_relevance_train = self.generate_dist_stats_feat("cosine", X_svd_train, dfTrain["id"].values, X_svd_train, dfTrain["id"].values, query_relevance_indices_dict,
                                                                                           dfTrain["qid"].values)
            with open("%s/train.%s_individual_svd%d_cosine_sim_stats_feat_by_relevance.feat.pkl" % (
                    path, feat_name, n_components), "wb") as f:
                cPickle.dump(cosine_sim_stats_feat_by_relevance_train, f, -1)
            with open("%s/train.%s_individual_svd%d_cosine_sim_stats_feat_by_query_relevance.feat.pkl" % (
                    path, feat_name, n_components), "wb") as f:
                cPickle.dump(cosine_sim_stats_feat_by_query_relevance_train, f, -1)
            ## test
            cosine_sim_stats_feat_by_relevance_test = self.generate_dist_stats_feat("cosine", X_svd_train, dfTrain["id"].values, X_svd_test, dfTest["id"].values, relevance_indices_dict)
            cosine_sim_stats_feat_by_query_relevance_test = self.generate_dist_stats_feat("cosine", X_svd_train, dfTrain["id"].values, X_svd_test, dfTest["id"].values, query_relevance_indices_dict, dfTest["qid"].values)
            with open("%s/%s.%s_individual_svd%d_cosine_sim_stats_feat_by_relevance.feat.pkl" % (
                    path, mode, feat_name, n_components), "wb") as f:
                cPickle.dump(cosine_sim_stats_feat_by_relevance_test, f, -1)
            with open("%s/%s.%s_individual_svd%d_cosine_sim_stats_feat_by_query_relevance.feat.pkl" % (
                    path, mode, feat_name, n_components), "wb") as f:
                cPickle.dump(cosine_sim_stats_feat_by_query_relevance_test, f, -1)

            ## update feat names
            new_feat_names.append(
                "%s_individual_svd%d_cosine_sim_stats_feat_by_relevance" % (feat_name, n_components))
            new_feat_names.append(
                "%s_individual_svd%d_cosine_sim_stats_feat_by_query_relevance" % (feat_name, n_components))

        return new_feat_names

    def create_vocabulary(dfTrain, vocabulary_type, vec_type, ngram_range):
        """
        根据vocabulary_type 生成
        :param vocabulary_type:common  individual
        :param vec_type:tfidf tfidf
        :return:
        """
        # 生成 vocabulary
        if vocabulary_type == "common":
            if vec_type == "tfidf":
                vec = getTFV(ngram_range=ngram_range)
            elif vec_type == "bow":
                vec = getBOW(ngram_range=ngram_range)
            vec.fit(dfTrain["all_text"])
            vocabulary = vec.vocabulary_
        elif vocabulary_type == "individual":
            vocabulary = None
        return vocabulary

    def gen_bow_tfidf_by_feat_column_names(self, path, dfTrain, dfTest, vec_type, mode, vocabulary, relevance_indices_dict, query_relevance_indices_dict, feat_names, column_names):
        """
        根据vec_type mode vocabulary 生成
        :param vec_type:'tfidf'/'bow'
        :param mode: 'valid' / 'test'
        :param vocabulary:
        :param relevance_indices_dict:
        :param query_relevance_indices_dict:
        :param feat_names:
        :prarm columns:
        :return:
        """
        new_feat_names = []
        for feat_name, column_name in zip(feat_names, column_names):
            # 根据 vec_type  bow/tfidf 生成不同的特征
            if vec_type == "tfidf":
                vec = getTFV(ngram_range=self.ngram_range, vocabulary=vocabulary)
            elif vec_type == "bow":
                vec = getBOW(ngram_range=self.ngram_range, vocabulary=vocabulary)
            X_train = vec.fit_transform(dfTrain[column_name])
            X_test = vec.transform(dfTest[column_name])
            # 生成basic bow tfidf 特征
            ##########################
            print "generate %s feat for %s" % (vec_type, column_name)

            with open("%s/train.%s.feat.pkl" % (path, feat_name), "wb") as f:
                cPickle.dump(X_train, f, -1)
            with open("%s/%s.%s.feat.pkl" % (path, mode, feat_name), "wb") as f:
                cPickle.dump(X_test, f, -1)

            if stats_feat_flag:
                feat_list = self.extract_bow_tfidf_cosine_sim_stats_feat(path, dfTrain, dfTest, feat_name, column_name, X_train, X_test, vec_type, mode, relevance_indices_dict, query_relevance_indices_dict)
                new_feat_names.extend(feat_list)
        return new_feat_names

    def gen_common_svd_by_feat_column_names(self, path, dfTrain, dfTest, X_vec_all_train, n_components, vec_type, mode, relevance_indices_dict, query_relevance_indices_dict, feat_names, column_names):
        """

        :param X_vec_all_train:
        :param n_components:
        :param mode:
        :param relevance_indices_dict:
        :param query_relevance_indices_dict:
        :return:
        """
        new_feat_names = []
        svd = TruncatedSVD(n_components=n_components, n_iter=15)
        svd.fit(X_vec_all_train)
        for feat_name, column_name in zip(feat_names, column_names):
            print "generate common %s-svd%d feat for %s" % (vec_type, n_components, column_name)
            # 生成common svd 特征
            with open("%s/train.%s.feat.pkl" % (path, feat_name), "rb") as f:
                X_vec_train = cPickle.load(f)
            with open("%s/%s.%s.feat.pkl" % (path, mode, feat_name), "rb") as f:
                X_vec_test = cPickle.load(f)
            X_svd_train = svd.transform(X_vec_train)
            X_svd_test = svd.transform(X_vec_test)
            with open("%s/train.%s_common_svd%d.feat.pkl" % (path, feat_name, n_components), "wb") as f:
                cPickle.dump(X_svd_train, f, -1)
            with open("%s/%s.%s_common_svd%d.feat.pkl" % (path, mode, feat_name, n_components), "wb") as f:
                cPickle.dump(X_svd_test, f, -1)
            ## update feat names
            new_feat_names.append("%s_common_svd%d" % (feat_name, n_components))

            if stats_feat_flag:
                #####################################
                ## bow/tfidf-svd cosine sim stats feat ##
                #####################################
                feat_list = self.extract_svd_cosine_sim_stats_feat(path, dfTrain, dfTest, feat_name, column_name, X_svd_train, X_svd_test, mode, n_components, relevance_indices_dict, query_relevance_indices_dict)
                new_feat_names.extend(feat_list)
        return new_feat_names

    def gen_individual_svd_by_feat_column_names(self, path, dfTrain, dfTest, n_components, vec_type, mode, relevance_indices_dict, query_relevance_indices_dict, feat_names, column_names):
        """
        generate individual svd feat
        :param n_components:
        :param mode:
        :param relevance_indices_dict:
        :param query_relevance_indices_dict:
        :return:
        """
        new_feat_names = []
        for feat_name, column_name in zip(feat_names, column_names):
            print "generate individual %s-svd%d feat for %s" % (vec_type, n_components, column_name)
            with open("%s/train.%s.feat.pkl" % (path, feat_name), "rb") as f:
                X_vec_train = cPickle.load(f)
            with open("%s/%s.%s.feat.pkl" % (path, mode, feat_name), "rb") as f:
                X_vec_test = cPickle.load(f)
            svd = TruncatedSVD(n_components=n_components, n_iter=15)
            X_svd_train = svd.fit_transform(X_vec_train)
            X_svd_test = svd.transform(X_vec_test)
            with open("%s/train.%s_individual_svd%d.feat.pkl" % (path, feat_name, n_components), "wb") as f:
                cPickle.dump(X_svd_train, f, -1)
            with open("%s/%s.%s_individual_svd%d.feat.pkl" % (path, mode, feat_name, n_components), "wb") as f:
                cPickle.dump(X_svd_test, f, -1)
            ## update feat names
            new_feat_names.append("%s_individual_svd%d" % (feat_name, n_components))

            if stats_feat_flag:
                #########################################
                ## bow/tfidf-svd cosine sim stats feat ##
                #########################################
                self.extract_svd_cosine_sim_stats_feat_individual(path, dfTrain, dfTest, feat_name, column_name, X_svd_train, X_svd_test, vec_type, mode,
                                                                  n_components, relevance_indices_dict, query_relevance_indices_dict, new_feat_names)
        return new_feat_names

    def extract_feat(self, path, dfTrain, dfTest, vec_type, mode, feat_names, column_names, vocabulary_type, svd_n_components):
        """
        extract all features
        1.fit a bow/tfidf on the all_text to get
        2.the common vocabulary to ensure query/title/description
        3.has the same length bow/tfidf for computing the similarity
        :param path:
        :param dfTrain:
        :param dfTest:
        :param mode:
        :param feat_names:
        :param column_names:
        :return:
        vocabulary_type:'common','dinididual'
        """
        # 保留最基本的三个特征： 'query_tfidf_common_vocabulary'，'title_tfidf_common_vocabulary'，'description_tfidf_common_vocabulary
        new_feat_names = copy(feat_names)
        # 找出所有的词汇
        vocabulary = self.create_vocabulary(vocabulary_type, vec_type)
        if stats_feat_flag:
            # 返回 类别为键，序号数组为值的字典
            relevance_indices_dict = self.get_sample_indices_by_relevance(dfTrain)
            # 返回 类别-qid为键，序号数组为值的字典
            query_relevance_indices_dict = self.get_sample_indices_by_relevance(dfTrain, "qid")

        feat_list = self.gen_bow_tfidf_by_feat_column_names(vec_type, mode, vocabulary, feat_names, column_names)
        new_feat_names.extend(feat_list)

        # cosine sim feat
        feat_list = self.extract_cosine_sim_feat(path, feat_names, mode)
        new_feat_names.extend(feat_list)

        # vstack 所有的feat
        for i, feat_name in enumerate(feat_names):
            with open("%s/train.%s.feat.pkl" % (path, feat_name), "rb") as f:
                X_vec_train = cPickle.load(f)
            if i == 0:
                X_vec_all_train = X_vec_train
            else:
                X_vec_all_train = vstack([X_vec_all_train, X_vec_train])

        for n_components in svd_n_components:
            feat_list = self.gen_common_svd_by_feat_column_names(dfTrain, dfTest, X_vec_all_train, n_components, vec_type, mode, relevance_indices_dict, query_relevance_indices_dict)
            new_feat_names.extend(feat_list)
            # cosine sim feat ##
            feat_list = self.extract_svd_cosine_sim_feat(path, feat_names, vec_type, mode, n_components)
            new_feat_names.extend(feat_list)

            feat_list = self.gen_individual_svd_by_feat_column_names(dfTrain, dfTest, n_components, vec_type, mode, relevance_indices_dict, query_relevance_indices_dict)
            new_feat_names.extend(feat_list)

        return new_feat_names

    def gen_basic_tfidf_feat(self):
        """
        入口函数
        :return:
        """
        # 读取训练数据和测试数据
        with open(config.processed_train_data_path, "rb") as f:
            df_train = cPickle.load(f)
        with open(config.processed_test_data_path, "rb") as f:
            df_test = cPickle.load(f)
        # 读取分层交叉验证数据索引
        with open("%s/stratifiedKFold.%s.pkl" % (config.data_folder, config.stratified_label), "rb") as f:
            skf = cPickle.load(f)


        # 合并三个列
        def cat_text(x):
            res = '%s %s %s' % (x['query'], x['product_title'], x['product_description'])
            return res

        # 添加一个all_test,保存所有文本
        df_train["all_text"] = list(df_train.apply(cat_text, axis=1))
        df_test["all_text"] = list(df_test.apply(cat_text, axis=1))

        # vec_type: "tfidf", "bow"] ; vocabulary_type："common"
        for vec_type in self.vec_types:
            ## save feat names
            feat_names = ["query", "title", "description"]
            feat_names = [name + "_%s_%s_vocabulary" % (vec_type, self.vocabulary_type) for name in feat_names]

            print("==================================================")
            print("Generate basic %s features..." % vec_type)

            print("For cross-validation...")
            for run in range(config.n_runs):
                # use 33% for training and 67 % for validation so we switch trainInd and validInd
                for fold, (validInd, trainInd) in enumerate(skf[run]):
                    print("Run: %d, Fold: %d" % (run + 1, fold + 1))
                    path = "%s/Run%d/Fold%d" % (config.feat_folder, run + 1, fold + 1)
                    dfTrain_train_train = df_train.iloc[trainInd].copy()
                    dfTrain_train_valid = df_train.iloc[validInd].copy()
                    self.extract_feat(path, dfTrain_train_train, dfTrain_train_valid, vec_type, "valid", feat_names, self.column_names)

            print("Done.")

            print("For training and testing...")
            path = "%s/All" % config.feat_folder
            ## extract feat
            feat_names = self.extract_feat(path, df_train, df_test, "test", feat_names, self.column_names)
            ## dump feat name
            ## file to save feat names
            feat_name_file = "%s/basic_%s_and_cosine_sim.feat_name" % (config.feat_folder, vec_type)
            self.dump_feat_name(feat_names, feat_name_file)

            print("All Done.")
