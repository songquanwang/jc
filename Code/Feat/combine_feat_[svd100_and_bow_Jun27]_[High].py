# coding:utf-8
"""
__file__
    
    combine_feat_[svd100_and_bow_Jun27]_[High].py

__description__

    This file generates one combination of feature set (High).
    Such features are used to generate the best single model with linear model, e.g., 
        - XGBoost linear booster with MSE objective
        - Sklearn Ridge

__author__

    Chenglong Chen < c.chenglong@gmail.com >

"""

import sys

sys.path.append("../")
from param_config import config
from gen_info import gen_info
from combine_feat import combine_feat, SimpleTransform

if __name__ == "__main__":
    feat_names = [

        # ## id feat
        ("qid", SimpleTransform()),

        ################
        ## Word count ##
        ################
        ('count_of_query_unigram', SimpleTransform(config.count_feat_transform)),
        ('count_of_unique_query_unigram', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_unique_query_unigram', SimpleTransform()),
        ('count_of_query_bigram', SimpleTransform(config.count_feat_transform)),
        ('count_of_unique_query_bigram', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_unique_query_bigram', SimpleTransform()),
        ('count_of_query_trigram', SimpleTransform(config.count_feat_transform)),
        ('count_of_unique_query_trigram', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_unique_query_trigram', SimpleTransform()),
        ('count_of_digit_in_query', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_digit_in_query', SimpleTransform()),
        ('count_of_title_unigram', SimpleTransform(config.count_feat_transform)),
        ('count_of_unique_title_unigram', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_unique_title_unigram', SimpleTransform()),
        ('count_of_title_bigram', SimpleTransform(config.count_feat_transform)),
        ('count_of_unique_title_bigram', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_unique_title_bigram', SimpleTransform()),
        ('count_of_title_trigram', SimpleTransform(config.count_feat_transform)),
        ('count_of_unique_title_trigram', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_unique_title_trigram', SimpleTransform()),
        ('count_of_digit_in_title', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_digit_in_title', SimpleTransform()),
        ('count_of_description_unigram', SimpleTransform(config.count_feat_transform)),
        ('count_of_unique_description_unigram', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_unique_description_unigram', SimpleTransform()),
        ('count_of_description_bigram', SimpleTransform(config.count_feat_transform)),
        ('count_of_unique_description_bigram', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_unique_description_bigram', SimpleTransform()),
        ('count_of_description_trigram', SimpleTransform(config.count_feat_transform)),
        ('count_of_unique_description_trigram', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_unique_description_trigram', SimpleTransform()),
        ('count_of_digit_in_description', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_digit_in_description', SimpleTransform()),
        ('count_of_query_unigram_in_title', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_query_unigram_in_title', SimpleTransform()),
        ('count_of_query_unigram_in_description', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_query_unigram_in_description', SimpleTransform()),
        ('count_of_title_unigram_in_query', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_title_unigram_in_query', SimpleTransform()),
        ('count_of_title_unigram_in_description', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_title_unigram_in_description', SimpleTransform()),
        ('count_of_description_unigram_in_query', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_description_unigram_in_query', SimpleTransform()),
        ('count_of_description_unigram_in_title', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_description_unigram_in_title', SimpleTransform()),
        ('title_unigram_in_query_div_query_unigram', SimpleTransform()),
        ('title_unigram_in_query_div_query_unigram_in_title', SimpleTransform()),
        ('description_unigram_in_query_div_query_unigram', SimpleTransform()),
        ('description_unigram_in_query_div_query_unigram_in_description', SimpleTransform()),
        ('count_of_query_bigram_in_title', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_query_bigram_in_title', SimpleTransform()),
        ('count_of_query_bigram_in_description', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_query_bigram_in_description', SimpleTransform()),
        ('count_of_title_bigram_in_query', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_title_bigram_in_query', SimpleTransform()),
        ('count_of_title_bigram_in_description', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_title_bigram_in_description', SimpleTransform()),
        ('count_of_description_bigram_in_query', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_description_bigram_in_query', SimpleTransform()),
        ('count_of_description_bigram_in_title', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_description_bigram_in_title', SimpleTransform()),
        ('title_bigram_in_query_div_query_bigram', SimpleTransform()),
        ('title_bigram_in_query_div_query_bigram_in_title', SimpleTransform()),
        ('description_bigram_in_query_div_query_bigram', SimpleTransform()),
        ('description_bigram_in_query_div_query_bigram_in_description', SimpleTransform()),
        ('count_of_query_trigram_in_title', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_query_trigram_in_title', SimpleTransform()),
        ('count_of_query_trigram_in_description', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_query_trigram_in_description', SimpleTransform()),
        ('count_of_title_trigram_in_query', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_title_trigram_in_query', SimpleTransform()),
        ('count_of_title_trigram_in_description', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_title_trigram_in_description', SimpleTransform()),
        ('count_of_description_trigram_in_query', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_description_trigram_in_query', SimpleTransform()),
        ('count_of_description_trigram_in_title', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_description_trigram_in_title', SimpleTransform()),
        ('title_trigram_in_query_div_query_trigram', SimpleTransform()),
        ('title_trigram_in_query_div_query_trigram_in_title', SimpleTransform()),
        ('description_trigram_in_query_div_query_trigram', SimpleTransform()),
        ('description_trigram_in_query_div_query_trigram_in_description', SimpleTransform()),

        ('pos_of_title_unigram_in_query_min', SimpleTransform(config.count_feat_transform)),
        ('pos_of_title_unigram_in_query_mean', SimpleTransform(config.count_feat_transform)),
        ('pos_of_title_unigram_in_query_median', SimpleTransform(config.count_feat_transform)),
        ('pos_of_title_unigram_in_query_max', SimpleTransform(config.count_feat_transform)),
        ('pos_of_title_unigram_in_query_std', SimpleTransform(config.count_feat_transform)),
        ('normalized_pos_of_title_unigram_in_query_min', SimpleTransform()),
        ('normalized_pos_of_title_unigram_in_query_mean', SimpleTransform()),
        ('normalized_pos_of_title_unigram_in_query_median', SimpleTransform()),
        ('normalized_pos_of_title_unigram_in_query_max', SimpleTransform()),
        ('normalized_pos_of_title_unigram_in_query_std', SimpleTransform()),
        ('pos_of_description_unigram_in_query_min', SimpleTransform(config.count_feat_transform)),
        ('pos_of_description_unigram_in_query_mean', SimpleTransform(config.count_feat_transform)),
        ('pos_of_description_unigram_in_query_median', SimpleTransform(config.count_feat_transform)),
        ('pos_of_description_unigram_in_query_max', SimpleTransform(config.count_feat_transform)),
        ('pos_of_description_unigram_in_query_std', SimpleTransform(config.count_feat_transform)),
        ('normalized_pos_of_description_unigram_in_query_min', SimpleTransform()),
        ('normalized_pos_of_description_unigram_in_query_mean', SimpleTransform()),
        ('normalized_pos_of_description_unigram_in_query_median', SimpleTransform()),
        ('normalized_pos_of_description_unigram_in_query_max', SimpleTransform()),
        ('normalized_pos_of_description_unigram_in_query_std', SimpleTransform()),
        ('pos_of_query_unigram_in_title_min', SimpleTransform(config.count_feat_transform)),
        ('pos_of_query_unigram_in_title_mean', SimpleTransform(config.count_feat_transform)),
        ('pos_of_query_unigram_in_title_median', SimpleTransform(config.count_feat_transform)),
        ('pos_of_query_unigram_in_title_max', SimpleTransform(config.count_feat_transform)),
        ('pos_of_query_unigram_in_title_std', SimpleTransform(config.count_feat_transform)),
        ('normalized_pos_of_query_unigram_in_title_min', SimpleTransform()),
        ('normalized_pos_of_query_unigram_in_title_mean', SimpleTransform()),
        ('normalized_pos_of_query_unigram_in_title_median', SimpleTransform()),
        ('normalized_pos_of_query_unigram_in_title_max', SimpleTransform()),
        ('normalized_pos_of_query_unigram_in_title_std', SimpleTransform()),
        ('pos_of_description_unigram_in_title_min', SimpleTransform(config.count_feat_transform)),
        ('pos_of_description_unigram_in_title_mean', SimpleTransform(config.count_feat_transform)),
        ('pos_of_description_unigram_in_title_median', SimpleTransform(config.count_feat_transform)),
        ('pos_of_description_unigram_in_title_max', SimpleTransform(config.count_feat_transform)),
        ('pos_of_description_unigram_in_title_std', SimpleTransform(config.count_feat_transform)),
        ('normalized_pos_of_description_unigram_in_title_min', SimpleTransform()),
        ('normalized_pos_of_description_unigram_in_title_mean', SimpleTransform()),
        ('normalized_pos_of_description_unigram_in_title_median', SimpleTransform()),
        ('normalized_pos_of_description_unigram_in_title_max', SimpleTransform()),
        ('normalized_pos_of_description_unigram_in_title_std', SimpleTransform()),
        ('pos_of_query_unigram_in_description_min', SimpleTransform(config.count_feat_transform)),
        ('pos_of_query_unigram_in_description_mean', SimpleTransform(config.count_feat_transform)),
        ('pos_of_query_unigram_in_description_median', SimpleTransform(config.count_feat_transform)),
        ('pos_of_query_unigram_in_description_max', SimpleTransform(config.count_feat_transform)),
        ('pos_of_query_unigram_in_description_std', SimpleTransform(config.count_feat_transform)),
        ('normalized_pos_of_query_unigram_in_description_min', SimpleTransform()),
        ('normalized_pos_of_query_unigram_in_description_mean', SimpleTransform()),
        ('normalized_pos_of_query_unigram_in_description_median', SimpleTransform()),
        ('normalized_pos_of_query_unigram_in_description_max', SimpleTransform()),
        ('normalized_pos_of_query_unigram_in_description_std', SimpleTransform()),
        ('pos_of_title_unigram_in_description_min', SimpleTransform(config.count_feat_transform)),
        ('pos_of_title_unigram_in_description_mean', SimpleTransform(config.count_feat_transform)),
        ('pos_of_title_unigram_in_description_median', SimpleTransform(config.count_feat_transform)),
        ('pos_of_title_unigram_in_description_max', SimpleTransform(config.count_feat_transform)),
        ('pos_of_title_unigram_in_description_std', SimpleTransform(config.count_feat_transform)),
        ('normalized_pos_of_title_unigram_in_description_min', SimpleTransform()),
        ('normalized_pos_of_title_unigram_in_description_mean', SimpleTransform()),
        ('normalized_pos_of_title_unigram_in_description_median', SimpleTransform()),
        ('normalized_pos_of_title_unigram_in_description_max', SimpleTransform()),
        ('normalized_pos_of_title_unigram_in_description_std', SimpleTransform()),
        # ('pos_of_title_bigram_in_query_min', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_title_bigram_in_query_mean', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_title_bigram_in_query_median', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_title_bigram_in_query_max', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_title_bigram_in_query_std', SimpleTransform(config.count_feat_transform)),
        # ('normalized_pos_of_title_bigram_in_query_min', SimpleTransform()),
        # ('normalized_pos_of_title_bigram_in_query_mean', SimpleTransform()),
        # ('normalized_pos_of_title_bigram_in_query_median', SimpleTransform()),
        # ('normalized_pos_of_title_bigram_in_query_max', SimpleTransform()),
        # ('normalized_pos_of_title_bigram_in_query_std', SimpleTransform()),
        # ('pos_of_description_bigram_in_query_min', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_description_bigram_in_query_mean', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_description_bigram_in_query_median', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_description_bigram_in_query_max', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_description_bigram_in_query_std', SimpleTransform(config.count_feat_transform)),
        # ('normalized_pos_of_description_bigram_in_query_min', SimpleTransform()),
        # ('normalized_pos_of_description_bigram_in_query_mean', SimpleTransform()),
        # ('normalized_pos_of_description_bigram_in_query_median', SimpleTransform()),
        # ('normalized_pos_of_description_bigram_in_query_max', SimpleTransform()),
        # ('normalized_pos_of_description_bigram_in_query_std', SimpleTransform()),
        # ('pos_of_query_bigram_in_title_min', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_query_bigram_in_title_mean', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_query_bigram_in_title_median', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_query_bigram_in_title_max', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_query_bigram_in_title_std', SimpleTransform(config.count_feat_transform)),
        # ('normalized_pos_of_query_bigram_in_title_min', SimpleTransform()),
        # ('normalized_pos_of_query_bigram_in_title_mean', SimpleTransform()),
        # ('normalized_pos_of_query_bigram_in_title_median', SimpleTransform()),
        # ('normalized_pos_of_query_bigram_in_title_max', SimpleTransform()),
        # ('normalized_pos_of_query_bigram_in_title_std', SimpleTransform()),
        # ('pos_of_description_bigram_in_title_min', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_description_bigram_in_title_mean', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_description_bigram_in_title_median', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_description_bigram_in_title_max', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_description_bigram_in_title_std', SimpleTransform(config.count_feat_transform)),
        # ('normalized_pos_of_description_bigram_in_title_min', SimpleTransform()),
        # ('normalized_pos_of_description_bigram_in_title_mean', SimpleTransform()),
        # ('normalized_pos_of_description_bigram_in_title_median', SimpleTransform()),
        # ('normalized_pos_of_description_bigram_in_title_max', SimpleTransform()),
        # ('normalized_pos_of_description_bigram_in_title_std', SimpleTransform()),
        # ('pos_of_query_bigram_in_description_min', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_query_bigram_in_description_mean', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_query_bigram_in_description_median', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_query_bigram_in_description_max', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_query_bigram_in_description_std', SimpleTransform(config.count_feat_transform)),
        # ('normalized_pos_of_query_bigram_in_description_min', SimpleTransform()),
        # ('normalized_pos_of_query_bigram_in_description_mean', SimpleTransform()),
        # ('normalized_pos_of_query_bigram_in_description_median', SimpleTransform()),
        # ('normalized_pos_of_query_bigram_in_description_max', SimpleTransform()),
        # ('normalized_pos_of_query_bigram_in_description_std', SimpleTransform()),
        # ('pos_of_title_bigram_in_description_min', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_title_bigram_in_description_mean', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_title_bigram_in_description_median', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_title_bigram_in_description_max', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_title_bigram_in_description_std', SimpleTransform(config.count_feat_transform)),
        # ('normalized_pos_of_title_bigram_in_description_min', SimpleTransform()),
        # ('normalized_pos_of_title_bigram_in_description_mean', SimpleTransform()),
        # ('normalized_pos_of_title_bigram_in_description_median', SimpleTransform()),
        # ('normalized_pos_of_title_bigram_in_description_max', SimpleTransform()),
        # ('normalized_pos_of_title_bigram_in_description_std', SimpleTransform()),
        # ('pos_of_title_trigram_in_query_min', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_title_trigram_in_query_mean', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_title_trigram_in_query_median', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_title_trigram_in_query_max', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_title_trigram_in_query_std', SimpleTransform(config.count_feat_transform)),
        # ('normalized_pos_of_title_trigram_in_query_min', SimpleTransform()),
        # ('normalized_pos_of_title_trigram_in_query_mean', SimpleTransform()),
        # ('normalized_pos_of_title_trigram_in_query_median', SimpleTransform()),
        # ('normalized_pos_of_title_trigram_in_query_max', SimpleTransform()),
        # ('normalized_pos_of_title_trigram_in_query_std', SimpleTransform()),
        # ('pos_of_description_trigram_in_query_min', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_description_trigram_in_query_mean', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_description_trigram_in_query_median', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_description_trigram_in_query_max', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_description_trigram_in_query_std', SimpleTransform(config.count_feat_transform)),
        # ('normalized_pos_of_description_trigram_in_query_min', SimpleTransform()),
        # ('normalized_pos_of_description_trigram_in_query_mean', SimpleTransform()),
        # ('normalized_pos_of_description_trigram_in_query_median', SimpleTransform()),
        # ('normalized_pos_of_description_trigram_in_query_max', SimpleTransform()),
        # ('normalized_pos_of_description_trigram_in_query_std', SimpleTransform()),
        # ('pos_of_query_trigram_in_title_min', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_query_trigram_in_title_mean', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_query_trigram_in_title_median', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_query_trigram_in_title_max', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_query_trigram_in_title_std', SimpleTransform(config.count_feat_transform)),
        # ('normalized_pos_of_query_trigram_in_title_min', SimpleTransform()),
        # ('normalized_pos_of_query_trigram_in_title_mean', SimpleTransform()),
        # ('normalized_pos_of_query_trigram_in_title_median', SimpleTransform()),
        # ('normalized_pos_of_query_trigram_in_title_max', SimpleTransform()),
        # ('normalized_pos_of_query_trigram_in_title_std', SimpleTransform()),
        # ('pos_of_description_trigram_in_title_min', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_description_trigram_in_title_mean', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_description_trigram_in_title_median', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_description_trigram_in_title_max', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_description_trigram_in_title_std', SimpleTransform(config.count_feat_transform)),
        # ('normalized_pos_of_description_trigram_in_title_min', SimpleTransform()),
        # ('normalized_pos_of_description_trigram_in_title_mean', SimpleTransform()),
        # ('normalized_pos_of_description_trigram_in_title_median', SimpleTransform()),
        # ('normalized_pos_of_description_trigram_in_title_max', SimpleTransform()),
        # ('normalized_pos_of_description_trigram_in_title_std', SimpleTransform()),
        # ('pos_of_query_trigram_in_description_min', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_query_trigram_in_description_mean', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_query_trigram_in_description_median', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_query_trigram_in_description_max', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_query_trigram_in_description_std', SimpleTransform(config.count_feat_transform)),
        # ('normalized_pos_of_query_trigram_in_description_min', SimpleTransform()),
        # ('normalized_pos_of_query_trigram_in_description_mean', SimpleTransform()),
        # ('normalized_pos_of_query_trigram_in_description_median', SimpleTransform()),
        # ('normalized_pos_of_query_trigram_in_description_max', SimpleTransform()),
        # ('normalized_pos_of_query_trigram_in_description_std', SimpleTransform()),
        # ('pos_of_title_trigram_in_description_min', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_title_trigram_in_description_mean', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_title_trigram_in_description_median', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_title_trigram_in_description_max', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_title_trigram_in_description_std', SimpleTransform(config.count_feat_transform)),
        # ('normalized_pos_of_title_trigram_in_description_min', SimpleTransform()),
        # ('normalized_pos_of_title_trigram_in_description_mean', SimpleTransform()),
        # ('normalized_pos_of_title_trigram_in_description_median', SimpleTransform()),
        # ('normalized_pos_of_title_trigram_in_description_max', SimpleTransform()),
        # ('normalized_pos_of_title_trigram_in_description_std', SimpleTransform()),

        ('description_missing', SimpleTransform()),

        ## jaccard coef
        ('jaccard_coef_of_unigram_between_query_title', SimpleTransform()),
        ('jaccard_coef_of_unigram_between_query_description', SimpleTransform()),
        ('jaccard_coef_of_unigram_between_title_description', SimpleTransform()),
        ('jaccard_coef_of_bigram_between_query_title', SimpleTransform()),
        ('jaccard_coef_of_bigram_between_query_description', SimpleTransform()),
        ('jaccard_coef_of_bigram_between_title_description', SimpleTransform()),
        ('jaccard_coef_of_trigram_between_query_title', SimpleTransform()),
        ('jaccard_coef_of_trigram_between_query_description', SimpleTransform()),
        ('jaccard_coef_of_trigram_between_title_description', SimpleTransform()),

        ## dice dist
        ('dice_dist_of_unigram_between_query_title', SimpleTransform()),
        ('dice_dist_of_unigram_between_query_description', SimpleTransform()),
        ('dice_dist_of_unigram_between_title_description', SimpleTransform()),
        ('dice_dist_of_bigram_between_query_title', SimpleTransform()),
        ('dice_dist_of_bigram_between_query_description', SimpleTransform()),
        ('dice_dist_of_bigram_between_title_description', SimpleTransform()),
        ('dice_dist_of_trigram_between_query_title', SimpleTransform()),
        ('dice_dist_of_trigram_between_query_description', SimpleTransform()),
        ('dice_dist_of_trigram_between_title_description', SimpleTransform()),

        #########
        ## BOW ##
        #########
        # ('query_bow_common_vocabulary', SimpleTransform()),
        # ('title_bow_common_vocabulary', SimpleTransform()),
        # ('description_bow_common_vocabulary', SimpleTransform()),
        ('title_bow_common_vocabulary_cosine_sim_stats_feat_by_relevance', SimpleTransform()),
        ('title_bow_common_vocabulary_cosine_sim_stats_feat_by_query_relevance', SimpleTransform()),
        # ('title_bow_common_vocabulary_cosine_sim_stats_feat_by_query_cat_relevance', SimpleTransform()),
        ('description_bow_common_vocabulary_cosine_sim_stats_feat_by_relevance', SimpleTransform()),
        ('description_bow_common_vocabulary_cosine_sim_stats_feat_by_query_relevance', SimpleTransform()),
        # ('description_bow_common_vocabulary_cosine_sim_stats_feat_by_query_cat_relevance', SimpleTransform()),
        ('query_bow_common_vocabulary_title_bow_common_vocabulary_bow_cosine_sim', SimpleTransform()),
        ('query_bow_common_vocabulary_description_bow_common_vocabulary_bow_cosine_sim', SimpleTransform()),
        ('title_bow_common_vocabulary_description_bow_common_vocabulary_bow_cosine_sim', SimpleTransform()),
        # # ('query_bow_common_vocabulary_common_svd100', SimpleTransform()),
        # # ('title_bow_common_vocabulary_common_svd100', SimpleTransform()),
        # # ('title_bow_common_vocabulary_common_svd100_cosine_sim_stats_feat_by_relevance', SimpleTransform()),
        # # ('title_bow_common_vocabulary_common_svd100_cosine_sim_stats_feat_by_query_relevance', SimpleTransform()),
        # # ('title_bow_common_vocabulary_common_svd100_cosine_sim_stats_feat_by_query_cat_relevance', SimpleTransform()),
        # # ('description_bow_common_vocabulary_common_svd100', SimpleTransform()),
        # # ('description_bow_common_vocabulary_common_svd100_cosine_sim_stats_feat_by_relevance', SimpleTransform()),
        # # ('description_bow_common_vocabulary_common_svd100_cosine_sim_stats_feat_by_query_relevance', SimpleTransform()),
        # # ('description_bow_common_vocabulary_common_svd100_cosine_sim_stats_feat_by_query_cat_relevance', SimpleTransform()),
        # # ('query_bow_common_vocabulary_title_bow_common_vocabulary_bow_common_svd100_cosine_sim', SimpleTransform()),
        # # ('query_bow_common_vocabulary_description_bow_common_vocabulary_bow_common_svd100_cosine_sim', SimpleTransform()),
        # # ('title_bow_common_vocabulary_description_bow_common_vocabulary_bow_common_svd100_cosine_sim', SimpleTransform()),
        # ('query_bow_common_vocabulary_individual_svd100', SimpleTransform()),
        # ('title_bow_common_vocabulary_individual_svd100', SimpleTransform()),
        # # ('title_bow_common_vocabulary_individual_svd100_cosine_sim_stats_feat_by_relevance', SimpleTransform()),
        # # ('title_bow_common_vocabulary_individual_svd100_cosine_sim_stats_feat_by_query_relevance', SimpleTransform()),
        # # ('title_bow_common_vocabulary_individual_svd100_cosine_sim_stats_feat_by_query_cat_relevance', SimpleTransform()),
        # ('description_bow_common_vocabulary_individual_svd100', SimpleTransform()),
        # # ('description_bow_common_vocabulary_individual_svd100_cosine_sim_stats_feat_by_relevance', SimpleTransform()),
        # # ('description_bow_common_vocabulary_individual_svd100_cosine_sim_stats_feat_by_query_relevance', SimpleTransform()),
        # # ('description_bow_common_vocabulary_individual_svd100_cosine_sim_stats_feat_by_query_cat_relevance', SimpleTransform()),

        ############
        ## TF-IDF ##
        ############
        ('query_tfidf_common_vocabulary', SimpleTransform()),
        ('title_tfidf_common_vocabulary', SimpleTransform()),
        ('description_tfidf_common_vocabulary', SimpleTransform()),
        ('title_tfidf_common_vocabulary_cosine_sim_stats_feat_by_relevance', SimpleTransform()),
        ('title_tfidf_common_vocabulary_cosine_sim_stats_feat_by_query_relevance', SimpleTransform()),
        # ('title_tfidf_common_vocabulary_cosine_sim_stats_feat_by_query_cat_relevance', SimpleTransform()),
        ('description_tfidf_common_vocabulary_cosine_sim_stats_feat_by_relevance', SimpleTransform()),
        ('description_tfidf_common_vocabulary_cosine_sim_stats_feat_by_query_relevance', SimpleTransform()),
        # ('description_tfidf_common_vocabulary_cosine_sim_stats_feat_by_query_cat_relevance', SimpleTransform()),
        ('query_tfidf_common_vocabulary_title_tfidf_common_vocabulary_tfidf_cosine_sim', SimpleTransform()),
        ('query_tfidf_common_vocabulary_description_tfidf_common_vocabulary_tfidf_cosine_sim', SimpleTransform()),
        ('title_tfidf_common_vocabulary_description_tfidf_common_vocabulary_tfidf_cosine_sim', SimpleTransform()),
        # ('query_tfidf_common_vocabulary_common_svd100', SimpleTransform()),
        # ('title_tfidf_common_vocabulary_common_svd100', SimpleTransform()),
        # ('title_tfidf_common_vocabulary_common_svd100_cosine_sim_stats_feat_by_relevance', SimpleTransform()),
        # ('title_tfidf_common_vocabulary_common_svd100_cosine_sim_stats_feat_by_query_relevance', SimpleTransform()),
        # ('title_tfidf_common_vocabulary_common_svd100_cosine_sim_stats_feat_by_query_cat_relevance', SimpleTransform()),
        # ('description_tfidf_common_vocabulary_common_svd100', SimpleTransform()),
        # ('description_tfidf_common_vocabulary_common_svd100_cosine_sim_stats_feat_by_relevance', SimpleTransform()),
        # ('description_tfidf_common_vocabulary_common_svd100_cosine_sim_stats_feat_by_query_relevance', SimpleTransform()),
        # ('description_tfidf_common_vocabulary_common_svd100_cosine_sim_stats_feat_by_query_cat_relevance', SimpleTransform()),
        # ('query_tfidf_common_vocabulary_title_tfidf_common_vocabulary_tfidf_common_svd100_cosine_sim', SimpleTransform()),
        # ('query_tfidf_common_vocabulary_description_tfidf_common_vocabulary_tfidf_common_svd100_cosine_sim', SimpleTransform()),
        # ('title_tfidf_common_vocabulary_description_tfidf_common_vocabulary_tfidf_common_svd100_cosine_sim', SimpleTransform()),
        # ('query_tfidf_common_vocabulary_individual_svd100', SimpleTransform()),
        # ('title_tfidf_common_vocabulary_individual_svd100', SimpleTransform()),
        # # ('title_tfidf_common_vocabulary_individual_svd100_cosine_sim_stats_feat_by_relevance', SimpleTransform()),
        # # ('title_tfidf_common_vocabulary_individual_svd100_cosine_sim_stats_feat_by_query_relevance', SimpleTransform()),
        # # ('title_tfidf_common_vocabulary_individual_svd100_cosine_sim_stats_feat_by_query_cat_relevance', SimpleTransform()),
        # ('description_tfidf_common_vocabulary_individual_svd100', SimpleTransform()),
        # # ('description_tfidf_common_vocabulary_individual_svd100_cosine_sim_stats_feat_by_relevance', SimpleTransform()),
        # # ('description_tfidf_common_vocabulary_individual_svd100_cosine_sim_stats_feat_by_query_relevance', SimpleTransform()),
        # # ('description_tfidf_common_vocabulary_individual_svd100_cosine_sim_stats_feat_by_query_cat_relevance', SimpleTransform()),

        #########################
        ## Cooccurrence TF-IDF ##
        #########################
        ('query_unigram_title_unigram_tfidf', SimpleTransform()),
        # ('query_unigram_title_bigram_tfidf', SimpleTransform()),
        ('query_unigram_description_unigram_tfidf', SimpleTransform()),
        # # ('query_unigram_description_bigram_tfidf', SimpleTransform()),
        ('query_bigram_title_unigram_tfidf', SimpleTransform()),
        # ('query_bigram_title_bigram_tfidf', SimpleTransform()),
        ('query_bigram_description_unigram_tfidf', SimpleTransform()),
        # # ('query_bigram_description_bigram_tfidf', SimpleTransform()),
        ('query_id_title_unigram_tfidf', SimpleTransform()),
        # ('query_id_title_bigram_tfidf', SimpleTransform()),
        ('query_id_description_unigram_tfidf', SimpleTransform()),
        # # ('query_id_description_bigram_tfidf', SimpleTransform()),
        # # ('query_cat_id_title_unigram_tfidf', SimpleTransform()),
        # # ('query_cat_id_title_bigram_tfidf', SimpleTransform()),
        # # ('query_cat_id_description_unigram_tfidf', SimpleTransform()),
        # # ('query_cat_id_description_bigram_tfidf', SimpleTransform()),
        # ('query_unigram_title_unigram_tfidf_individual_svd100', SimpleTransform()),
        # # ('query_unigram_title_bigram_tfidf_individual_svd100', SimpleTransform()),
        # # ('query_unigram_description_unigram_tfidf_individual_svd100', SimpleTransform()),
        # # ('query_unigram_description_bigram_tfidf_individual_svd100', SimpleTransform()),
        # ('query_bigram_title_unigram_tfidf_individual_svd100', SimpleTransform()),
        # # ('query_bigram_title_bigram_tfidf_individual_svd100', SimpleTransform()),
        # # ('query_bigram_description_unigram_tfidf_individual_svd100', SimpleTransform()),
        # # ('query_bigram_description_bigram_tfidf_individual_svd100', SimpleTransform()),
        # ('query_id_title_unigram_tfidf_individual_svd100', SimpleTransform()),
        # # ('query_id_title_bigram_tfidf_individual_svd100', SimpleTransform()),
        # # ('query_id_description_unigram_tfidf_individual_svd100', SimpleTransform()),
        # # ('query_id_description_bigram_tfidf_individual_svd100', SimpleTransform()),
        # # ('query_cat_id_title_unigram_tfidf_individual_svd100', SimpleTransform()),
        # # ('query_cat_id_title_bigram_tfidf_individual_svd100', SimpleTransform()),
        # # ('query_cat_id_description_unigram_tfidf_individual_svd100', SimpleTransform()),
        # # ('query_cat_id_description_bigram_tfidf_individual_svd100', SimpleTransform()),

    ]

    gen_info(feat_path_name="svd100_and_bow_Jun27")
    combine_feat(feat_names, feat_path_name="svd100_and_bow_Jun27")
