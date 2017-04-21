# coding=utf-8
__author__ = 'songquanwang'

from pyspark import StorageLevel, SparkContext, SparkConf

import csv
from itertools import islice

from  jd.dao.sku_attr_cate import SkuAttrCateDao
from jd.dao.user import UserDao
from jd.dao.user_sku_action_dao import UserSkuActionDao
from jd.dao.user_sku_comment_dao import UserSkuCommentDao
import jd.utils.base_utils as utils

sc = SparkContext(conf=SparkConf().setAppName("jc"))
sku_dao = SkuAttrCateDao()
user_dao = UserDao()
user_sku_action_dao = UserSkuActionDao()
user_sku_comment_dao = UserSkuCommentDao()


def read_csv(file_path, start_line, len):
    """
    从start_line 读取len行数据返回 很慢
    :param file_path:
    :param start_line:
    :param len:
    :return:
    """
    try:
        rows = []
        with open(file_path) as csvfile:
            csv_reader = csv.reader(csvfile)
            next(islice(csv_reader, start_line, start_line), None)
            # for i, row in enumerate(csv_reader):
            #     if i == len:
            #         break
            #     rows.append(row)
            rows = [row for i, row in enumerate(csv_reader) if i < len]
    finally:
        csvfile.close()
    return rows


def merge_data():
    user_map = sc.broadcast(user_dao.get_data().map(lambda line: (line[0], line[1:])).collectAsMap())

    sku_map = sku_dao.get_data().map(lambda line: (line[0], line[1:])).collectAsMap()

    def sort_by_date(values, key_index):
        return sorted(values, key=lambda x: x[key_index])

    comment_map = user_sku_comment_dao.get_data().map(lambda line: (line[1], line[0] + line[2:])).groupByKey().mapValues(lambda values: sort_by_date(values, 0)).collectAsMap()

    user_broad = sc.broadcast(user_map)
    sku_broad = sc.broadcast(sku_map)
    comment_broad = sc.broadcast(comment_map)

    rdd_action_0301_0315 = user_sku_action_dao.get_data_1().map(lambda line: ((line[0], line[1]), line[2:5])).groupByKey().mapValues(lambda values: sort_by_date(values, 0))
    rdd_action_0316_0331 = user_sku_action_dao.get_data_2().map(lambda line: ((line[0], line[1]), line[2:5])).groupByKey().mapValues(lambda values: sort_by_date(values, 0))
    rdd_action_0401_0415 = user_sku_action_dao.get_data_3().map(lambda line: ((line[0], line[1]), line[2:5])).groupByKey().mapValues(lambda values: sort_by_date(values, 0))

    def merge_data(value):
        user_id = value[0]
        sku_id = value[1]
        user_m = user_broad.value[user_id]
        sku_m = sku_broad.value[sku_id]
        comment_m = comment_broad.value[sku_id]
        row = [user_id] + user_m + [sku_id] + sku_m + [value] + [comment_m]
        return row

    merge_data_rdd_1 = rdd_action_0301_0315.mapValues(merge_data)
    merge_data_rdd_2 = rdd_action_0316_0331.mapValues(merge_data)
    merge_data_rdd_3 = rdd_action_0401_0415.mapValues(merge_data)
    merge_path_1 = 'hdfs://172.22.100.100:8020/user/mart_dm_tbi/app.db/s_jc/merge_data_0301_0315'
    merge_path_2 = 'hdfs://172.22.100.100:8020/user/mart_dm_tbi/app.db/s_jc/merge_data_0316_0331'
    merge_path_3 = 'hdfs://172.22.100.100:8020/user/mart_dm_tbi/app.db/s_jc/merge_data_0401_0415'
    utils.save_text_rdd(merge_path_1, merge_data_rdd_1)
    utils.save_text_rdd(merge_path_2, merge_data_rdd_2)
    utils.save_text_rdd(merge_path_3, merge_data_rdd_3)

    def test():
        file_path = '../data/JData_Action_0301_0315.csv'
        csv_reader = csv.reader(open(file_path))

        action_0301_0315 = read_csv('D:/github/Kaggle_CrowdFlower/jd/data/JData_Action_0301_0315.csv', 0, 1)
        action_0316_0331 = read_csv('../data/JData_Action_0316_0331.csv', 0, 10)
        action_0401_0315 = read_csv('../data/JData_Action_0401_0415.csv', 0, 10)
        comments = read_csv('../data/JData_Comment.csv', 0, 100)
        products = read_csv('../data/JData_Product.csv', 12477, 100)
        users = read_csv('../data/JData_User.csv', 0, 100)
