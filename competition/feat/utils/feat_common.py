# coding:utf-8
__author__ = 'songquanwang'

from competition.feat.nlp import ngram
from competition.feat.nlp.nlp_utils import preprocess_data


def gen_temp_feat(df):
    """
    用户组合其他特征的临时特征，这些基本特征会用到
    :param df:
    :return:
    """
    ## unigram
    print "generate unigram"
    df["query_unigram"] = list(df.apply(lambda x: preprocess_data(x["query"]), axis=1))
    df["title_unigram"] = list(df.apply(lambda x: preprocess_data(x["product_title"]), axis=1))
    df["description_unigram"] = list(df.apply(lambda x: preprocess_data(x["product_description"]), axis=1))
    ## bigram
    print "generate bigram"
    join_str = "_"
    df["query_bigram"] = list(df.apply(lambda x: ngram.getBigram(x["query_unigram"], join_str), axis=1))
    df["title_bigram"] = list(df.apply(lambda x: ngram.getBigram(x["title_unigram"], join_str), axis=1))
    df["description_bigram"] = list(df.apply(lambda x: ngram.getBigram(x["description_unigram"], join_str), axis=1))
    ## trigram
    print "generate trigram"
    join_str = "_"
    df["query_trigram"] = list(df.apply(lambda x: ngram.getTrigram(x["query_unigram"], join_str), axis=1))
    df["title_trigram"] = list(df.apply(lambda x: ngram.getTrigram(x["title_unigram"], join_str), axis=1))
    df["description_trigram"] = list(df.apply(lambda x: ngram.getTrigram(x["description_unigram"], join_str), axis=1))
    return df
