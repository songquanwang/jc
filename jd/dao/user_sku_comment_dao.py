# -*- coding: utf-8 -*-
__author__ = 'songquanwang'

import jd.utils.base_utils as utils


class UserSkuCommentDao():
    """
    评价数据
    对应文件：JData_Comment.cvs
    行数：558552 行
    主键：
    dt：（dt,sku_id)
    [u'2016-02-01', u'2016-02-08', u'2016-02-15', u'2016-02-22', u'2016-02-29', u'2016-03-07',
    u'2016-03-14', u'2016-03-21', u'2016-03-28', u'2016-04-04', u'2016-04-11', u'2016-04-15'] 12天
    sku_id: 46546个不同的sku  共：558552 条

    """
    _path1 = 'hdfs://172.22.100.100:8020/user/mart_dm_tbi/app.db/s_jc/user_comment'

    def __init__(self, sc, filter=utils.default_filter):
        self.sc = sc
        self.filter = filter

    def get_name(self):
        """
        获取表名称
        :return:
        """
        return "use_comment"

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
        ['2016-02-01', '1000', '3', '1', '0.0417']
        :return:
        """
        schema = utils.init_table([

            'dt',  # 截止到时间	 粒度到天
            'sku_id',  # 商品编号	 脱敏
            'comment_num',  # 累计评论数  分段	 0表示无评论，1表示有1条评论， 2表示有2-10条评论， 3表示有11-50条评论， 4表示大于50条评论
            'has_bad_comment',  # 是否有差评	 0表示无，1表示有
            'bad_comment_rate'  # 差评率	 差评数占总评论数的比重

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
