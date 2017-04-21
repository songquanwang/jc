# -*- coding: utf-8 -*-
__author__ = 'songquanwang'

import jd.utils.base_utils as utils


class UserDao():
    """
    用户数据
    文本文件：JData_User.csv
    行数：232741 行
    主键：user_id
    """
    _path1 = 'hdfs://172.22.100.100:8020/user/mart_dm_tbi/app.db/s_jc/user'

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
        1,-1,0,5,2004-10-12
        :return:
        """
        schema = utils.init_table([

            'user_id',  # 用户ID	 脱敏
            'age',  # 年龄段	 -1表示未知
            'sex',  # 性别	 0表示男，1表示女，2表示保密
            'user_lv_cd',  # 用户等级	 有顺序的级别枚举，越高级别数字越大
            'user_reg_dt'  # 用户注册日期	 粒度到天

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

    def save_data(self, dt, path, rdd):
        """
        保存rdd 到指定分区和文件，rdd要按照 schema格式
        :return:
        """
        return
