# -*- coding: utf-8 -*-
import numpy as np

num_of_class = 4
bootstrap_ratio = 1
bootstrap_replacement = False
bagging_size = 1

ebc_hard_threshold = False
verbose_level = 1


# 模型算法包路径 libfm rgf
libfm_exe = "../../libfm-1.40.windows/libfm.exe"
call_exe = "../../rgf1.2/test/call_exe.pl"
rgf_exe = "../../rgf1.2/bin/rgf.exe"

## cv交叉验证配置
n_runs = 3
n_folds = 3
stratified_label = "query"

# 路径配置
data_folder = "../../Data"
feat_folder = "../../Feat/solution"
original_train_data_path = "%s/train.csv" % data_folder
original_test_data_path = "%s/test.csv" % data_folder
processed_train_data_path = "%s/train.processed.csv.pkl" % feat_folder
processed_test_data_path = "%s/test.processed.csv.pkl" % feat_folder
pos_tagged_train_data_path = "%s/train.pos_tagged.csv.pkl" % feat_folder
pos_tagged_test_data_path = "%s/test.pos_tagged.csv.pkl" % feat_folder

output_path = "../../Output"

# nlp related
drop_html_flag = True
basic_tfidf_ngram_range = (1, 3)
basic_tfidf_vocabulary_type = "common"
cooccurrence_tfidf_ngram_range = (1, 1)
cooccurrence_word_exclude_stopword = False
stemmer_type = "porter"  # "snowball"

# transform for count features
count_feat_transform = np.sqrt
