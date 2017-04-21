# -*- coding: utf-8 -*-
__author__ = 'songquanwang'

import jd.utils.base_utils as utils


class SkuAttrCateDao():
    """
    商品数据
    文本文件:JData_Product.csv
    行数：12477行
    主键：sku_id c
    cate数据：只有一条 ‘6'
    brand数目：165个brand
    """

    _path1 = 'hdfs://172.22.100.100:8020/user/mart_dm_tbi/app.db/s_jc/sku_attr_cate'

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
        ['89903', '1', '3', '-1', '6', '427']
        :return:
        """
        schema = utils.init_table([

            'sku_id',  # 商品编号	 脱敏
            'attr1',  # 属性1	 枚举，-1表示未知
            'attr2',  # 属性2	 枚举，-1表示未知
            'attr3',  # 属性3	 枚举，-1表示未知
            'cate',  # 品类ID	 脱敏
            'brand'  # 品牌ID	 脱敏

        ])
        return schema

    def get_data(self):
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
