# -*- coding: utf-8 -*-
__author__ = 'songquanwang'

import jd.utils.base_utils as utils


class UserSkuActionDao():
    # 行为数据
    # 47139391 行
    _path1 = 'hdfs://172.22.100.100:8020/user/mart_dm_tbi/app.db/s_jc/user_sku_0301_0315'
    # 41305822 行
    _path2 = 'hdfs://172.22.100.100:8020/user/mart_dm_tbi/app.db/s_jc/user_sku_0316_0331'
    # 42925679 行
    _path3 = 'hdfs://172.22.100.100:8020/user/mart_dm_tbi/app.db/s_jc/user_sku_0401_0415'

    def __init__(self, sc, filter=utils.default_filter):
        self.sc = sc
        self.filter = filter

    def get_name(self):
        """
        获取表名称
        :return:
        """
        return "user sku dao"

    def get_seperator(self):
        """
        获取表字段发分隔符
        :return:
        """
        return ","

    def get_schema(self):
        """
        获取表定义，返回 字段名:序号 dict {'字段名1':1,'字段名2':2,...,'字段名N':N}
        数据例子：
        178555,102226,2016-03-08 01:21:50,,1,6,235
        140483,150146,2016-03-08 01:21:57,217,6,9,78

        :return:
        """
        schema = utils.init_table([

            'user_id',  # 用户编号	 脱敏
            'sku_id',  # 商品编号	 脱敏
            'time',  # 行为时间
            'model_id',  # 点击模块编号，如果是点击	 脱敏
            'type',  # 1.浏览（指浏览商品详情页）； 2.加入购物车；3.购物车删除；4.下单；5.关注；6.点击
            'cate',  # 品类ID	 脱敏
            'brand',  # 品牌ID	 脱敏

        ])
        return schema

    def get_data_1(self):
        """
        获取表某个dt数据，或者某个dt下某个path数据,经过split 切分
        :return:
        """
        seperator = self.get_seperator()
        data_rdd = self.sc.textFile(self._path1).map(lambda line: line.split(seperator))
        return self.filter(data_rdd)

    def get_data_2(self):
        """
        获取表某个dt数据，或者某个dt下某个path数据,经过split 切分
        :return:
        """
        seperator = self.get_seperator()
        data_rdd = self.sc.textFile(self._path2).map(lambda line: line.split(seperator))
        return self.filter(data_rdd)

    def get_data_3(self):
        """
        获取表某个dt数据，或者某个dt下某个path数据,经过split 切分
        :return:
        """
        seperator = self.get_seperator()
        data_rdd = self.sc.textFile(self._path3).map(lambda line: line.split(seperator))
        return self.filter(data_rdd)

    def save_data(self, dt, path, rdd):
        """
        保存rdd 到指定分区和文件，rdd要按照 schema格式
        :return:
        """
        return
