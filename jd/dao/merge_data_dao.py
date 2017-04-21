# -*- coding: utf-8 -*-
__author__ = 'songquanwang'

import jd.utils.base_utils as utils


class MergeDataDao():
    # merge数据
    #  行
    _path1 = 'hdfs://172.22.100.100:8020/user/mart_dm_tbi/app.db/s_jc/merge_data_0301_0315'
    #  行
    _path2 = 'hdfs://172.22.100.100:8020/user/mart_dm_tbi/app.db/s_jc/merge_data_0316_0331'
    #  行
    _path3 = 'hdfs://172.22.100.100:8020/user/mart_dm_tbi/app.db/s_jc/merge_data_0401_0415'

    def __init__(self, sc, filter=utils.default_filter):
        self.sc = sc
        self.filter = filter

    def get_name(self):
        """
        获取表名称
        :return:
        """
        return "merge data dao"

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

            # 用户信息
            'user_id',  # 用户ID	 脱敏
            'age',  # 年龄段	 -1表示未知
            'sex',  # 性别	 0表示男，1表示女，2表示保密
            'user_lv_cd',  # 用户等级	 有顺序的级别枚举，越高级别数字越大
            'user_reg_dt'  # 用户注册日期	 粒度到天
            # 商品信息
            'sku_id',  # 商品编号	 脱敏
            'attr1',  # 属性1	 枚举，-1表示未知
            'attr2',  # 属性2	 枚举，-1表示未知
            'attr3',  # 属性3	 枚举，-1表示未知
            'cate',  # 品类ID	 脱敏
            'brand',  # 品牌ID	 脱敏
            'action_list',  # 行为信息 （按照时间顺序）[(time,model_id,type)]
            'comment_list',  # 评论信息 [(截止时间,累计评论数,是否有差评,差评率)] -- 按照时间顺序 （该属性与产品对应）

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
