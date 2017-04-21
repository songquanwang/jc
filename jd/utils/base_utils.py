# -*- coding: utf-8 -*-
__author__ = 'songquanwang'

import os
import datetime as dt
import math
from collections import OrderedDict
import numbers
import six
from itertools import chain
from dateutil.relativedelta import relativedelta


# 经典bug sales = [v == -1 and 0 or v for k, v in dt_dict.items()]
# 注意 and 0 后 会执行 v


def mean(values):
    """
    模拟numpy mean 空数组返回 0.0
    :param values:
    :return:
    """
    if isinstance(values, numbers.Number):
        return float(values)
    result = 0.0
    if len(values) > 0:
        result = float(sum(values)) / len(values)
    return result


def vector_add(v1, v2):
    """
    模拟两个向量相加 :注意：不支持长度不一样的向量
    [1,2,3] + [2,4,5] =[3,6,8]
    v1 =[[1, 2], [4, 7]]
    v2 = [[11, 5], [21, 4]]

    v1 =[1,2,3]
    v2= [2,4,5]

    v1=100
    v2=200
    :param v1:
    :param v2:
    :return:
    """
    if isinstance(v1, list) and isinstance(v2, list):
        return [vector_add(x1, x2) for x1, x2 in zip(v1, v2)]
    else:
        return sum([v1, v2])


def argmax(arr):
    """
    模拟numpy argmax
    :param arr:
    :return:
    """
    if len(arr) == 0:
        raise ValueError("paremeter error!")
    m = max(arr)
    ret = -1
    for i, v in enumerate(arr):
        if v == m:
            ret = i
            break
    return ret


def mapd(r, p):
    """
    如果分母为零 返回 分子（误差）
    :param r:
    :param p:
    :return:
    """
    return r > 0 and abs(p - r) / r or abs(p - r)


def diff_mapd(r, diff):
    """
    如果分母为零 返回 分子（误差）
    :param r:
    :param p:
    :return:
    """
    return r > 0 and abs(diff) / r or abs(diff)


def mapd_list(real_list, predict_list):
    """
    求两个数组的mapd (一段时间真实销量和一段时间预测销量）;能够处理None ，nan 改成0；多个周期误差不累计
    :param real_list:
    :param predict_list:
    :return:
    """
    real_sales = map(lambda e: (e is not None and not math.isnan(e)) and e or 0, real_list)
    predict_sales = map(lambda e: (e is not None and not math.isnan(e)) and e or 0, predict_list)
    return mapd(sum(real_sales), sum(predict_sales))


def mapd_list_acc(real_list, predict_list):
    """
    求两个数组的mapd (多个周期误差累计）;能够处理None ，nan 改成0
    :param real_list:
    :param predict_list:
    :return:
    """
    real_sales = map(lambda e: (e is not None and not math.isnan(e)) and e or 0, real_list)
    predict_sales = map(lambda e: (e is not None and not math.isnan(e)) and e or 0, predict_list)
    diff = [abs(x - y) for x, y in zip(predict_sales, real_sales)]
    real_sum = sum(real_sales)
    diff_sum = sum(diff)
    return real_sum > 0 and diff_sum / float(real_sum) or diff_sum


def diff_sum(real_list, predict_list):
    """
    返回两个补全后的数组的差值
    :param real_list:    如果为None 会抛出异常
    :param predict_list: 如果为None 会抛出异常
    :return:
    """
    real_sales = map(lambda e: (e is not None and not math.isnan(e)) and e or 0, real_list)
    predict_sales = map(lambda e: (e is not None and not math.isnan(e)) and e or 0, predict_list)
    diff_sales = [abs(x - y) for x, y in zip(predict_sales, real_sales)]
    return sum(diff_sales)


def diff_square_sum(real_list, predict_list):
    """
    返回两个补全后的数组的差值 平方和
    :param real_list:    如果为None 会抛出异常
    :param predict_list: 如果为None 会抛出异常
    :return:
    """
    real_sales = map(lambda e: (e is not None and not math.isnan(e)) and e or 0, real_list)
    predict_sales = map(lambda e: (e is not None and not math.isnan(e)) and e or 0, predict_list)
    diff_sales = [(x - y) ** 2 for x, y in zip(predict_sales, real_sales)]
    return sum(diff_sales)


def rmse_list(real_list, predict_list):
    """
    求两个数组的rmse(与网上的rmse不同，分母是真实销量和) (一段时间真实销量和一段时间预测销量）;能够处理None ，nan 改成0
    :param real_list:
    :param predict_list:
    :return:
    """
    real_sales = map(lambda e: (e is not None and not math.isnan(e)) and e or 0, real_list)
    predict_sales = map(lambda e: (e is not None and not math.isnan(e)) and e or 0, predict_list)
    real_sum = sum(predict_sales)
    diff_sum = diff_square_sum(real_sales, predict_sales)
    return real_sum > 0 and math.sqrt(diff_sum / float(real_sum)) or diff_sum


def sum_list(data_list):
    """
    求两个数组的和;能够处理None ，nan 改成0
    :param data_list:
    :return:
    """
    return sum([float(e) for e in data_list if e is not None and e != 'nan'])


def day_to_period(date_sales, start_date, days, period, period_number, step, miss_handle=1):
    """
    传入每天销量数组，求出指定周期长度的销量
    1. 验证参数,参数数量不符会抛出异常
    2. 一天多个销量相加
    3. 'nan',None异常值处理 ，跟缺失值一样处理
    4. 滑动求和
    5. miss_handle:缺失值处理
        1>有任何一天销量缺失返回 None
        2>对缺失销量补 0
        3>对缺失销量补 mean
        4>对缺失销量使用临近值填充
    例子:
    date_sales=[('2016-05-02',2),('2016-05-01',1),('2016-05-03',3),('2016-05-04',4),('2016-05-05',5),('2016-05-06',6),('2016-05-01',1),('2016-05-08',8),
    ('2016-05-09',9),('2016-05-02',2),('2016-05-11',11)]
    start_date=dt.datetime.strptime('2016-05-01', "%Y-%m-%d").date()
    days=11
    period=7
    period_number=3
    step =2
    :param date_sales:
    :param start_date:
    :param days:
    :param period:
    :param period_number:
    :param miss_handle
    :return:
    """

    if period_number != (days - period) / step + 1:
        raise ValueError("paremeter error!")
    dt_list = [(start_date + dt.timedelta(d)).strftime('%Y-%m-%d') for d in range(0, days)]
    # -1 代表不存在
    dt_dict = OrderedDict(zip(dt_list, [-1] * days))
    # 如果date_sales有时间段内没有的日期，自动忽略
    # 会累加重复天数
    # 'nan' None按缺失处理
    for sale in date_sales:
        if dt_dict.has_key(sale[0]) and sale[1] is not None and not math.isnan(float(sale[1])):
            if dt_dict[sale[0]] == -1:
                dt_dict[sale[0]] = 0
            dt_dict[sale[0]] += sale[1]
    # 缺失值
    not_exist_num = [s for s in dt_dict.values() if s == -1]
    # 有缺失值返回 None
    if miss_handle == 1 and len(not_exist_num) > 0:
        return None
    # 有缺失值补零 注意：如果v ==-1 and 0 or v 就会永远执行 v **
    if miss_handle == 2:
        sales = [v != -1 and v or 0 for k, v in dt_dict.items()]
    elif miss_handle == 3:
        # 有缺失值补充 均值
        non_empty = [s for s in dt_dict.values() if s != -1]
        default_value = mean(non_empty)
        # v == -1是用均值替换
        # 注意：v == -1 and default_value  有bug，如果均值未0 会 执行 v 弄成-1
        # v != -1 and v or default_value   有bug :v == 0 时会用 default_value替换
        sales = []
        for k, v in dt_dict.items():
            if v == -1:
                sales.append(default_value)
            else:
                sales.append(v)
    else:
        # 按就近填充值
        pre = -1
        filled_begin = False
        sales = dt_dict.values()
        for i, v in enumerate(sales):
            if v != -1:
                if not filled_begin:
                    sales[0:i] = [v] * i
                    filled_begin = True
                pre = i
            else:
                sales[i] = sales[pre]
    # 滑动求和 最后一个周期会把剩余部分求和
    period_list = [sum(sales[d:d + period]) for d in range(0, period_number * step, step)]
    return period_list


def month_to_period(date_sales, start_date, months, period, period_number, step, miss_handle=1):
    """
    传入每天销量数组，求出指定周期长度的销量
    1. 验证参数,参数数量不符会抛出异常
    2. 多个多个销量相加
    3. 'nan',None异常值处理 ，跟缺失值一样处理
    4. 滑动求和
    5. miss_handle:缺失值处理
        1>有任何一天销量缺失返回 None
        2>对缺失销量补 0
        3>对缺失销量补 mean
        4>对缺失销量使用临近值填充
    例子:
    date_sales=[('2016-02',2),('2016-01',1),('2016-03',3),('2016-04',4),('2016-05',5),('2016-06',6),('2016-01',1),('2016-08',8),
    ('2016-09',9),('2016-02',2),('2016-11',11)]
    start_date=dt.datetime.strptime('2016-01', "%Y-%m").date()
    months=11
    period=7
    period_number=3
    step =2
    :param date_sales:
    :param start_date:
    :param months:
    :param period:
    :param period_number:
    :param miss_handle
    :return:
    """
    if period_number != (months - period) / step + 1:
        raise ValueError("paremeter error!")
    dt_list = [(start_date + relativedelta(months=m)).strftime('%Y-%m') for m in range(0, months)]
    # -1 代表不存在
    dt_dict = OrderedDict(zip(dt_list, [-1] * months))
    # 如果date_sales有时间段内没有的日期，自动忽略
    # 会累加重复天数
    # 'nan' None按缺失处理
    for sale in date_sales:
        if dt_dict.has_key(sale[0]) and sale[1] is not None and not math.isnan(float(sale[1])):
            if dt_dict[sale[0]] == -1:
                dt_dict[sale[0]] = 0
            dt_dict[sale[0]] += sale[1]
    # 缺失值
    not_exist_num = [s for s in dt_dict.values() if s == -1]
    # 有缺失值返回 None
    if miss_handle == 1 and len(not_exist_num) > 0:
        return None
    # 有缺失值补零 注意：如果v ==-1 and 0 or v 就会永远执行 v **
    if miss_handle == 2:
        sales = [v != -1 and v or 0 for k, v in dt_dict.items()]
    elif miss_handle == 3:
        # 有缺失值补充 均值
        non_empty = [s for s in dt_dict.values() if s != -1]
        default_value = mean(non_empty)
        # v == -1是用均值替换
        # 注意：v == -1 and default_value  有bug，如果均值未0 会 执行 v 弄成-1
        # v != -1 and v or default_value   有bug :v == 0 时会用 default_value替换
        sales = []
        for k, v in dt_dict.items():
            if v == -1:
                sales.append(default_value)
            else:
                sales.append(v)
    else:
        # 按就近填充值
        pre = -1
        filled_begin = False
        sales = dt_dict.values()
        for i, v in enumerate(sales):
            if v != -1:
                if not filled_begin:
                    sales[0:i] = [v] * i
                    filled_begin = True
                pre = i
            else:
                sales[i] = sales[pre]
    # 滑动求和
    period_list = [sum(sales[d:d + period]) for d in range(0, period_number * step, step)]
    return period_list


def to_period(from_pd, to_pd, pd_number, data_list):
    """
    N:M周期转换
    nan,None替换成0
    不足补0
    结果保证 pd_number个
    [1,2,3,4,5,6,7,8,9,10,11,12,13,14] 1-7 [28,77]
    [7,14,21,28]   7-1 [1,1,1,1,1,1,1,2,2,2,2,2,2,2,3,3,3,3,3,3,3,4,4,4,4,4,4,4]
    :param from_pd: 原来什么单位
    :param to_pd:   转成什么单位
    :param predict_list: 要转换的数据[]
    :return:
    """
    # data_list = [float(d) for d in data_list]
    # data_list  可能有 'nan' [[u'1996321', u'682', u'nan', u'nan', u'nan', u'nan', u'-0.5', u'0.0']]
    data_list_processed = []
    for e in data_list:
        if e is None or math.isnan(float(e)):
            f = 0
        else:
            f = float(e)
        data_list_processed.append(f)
    # 如果周期不变，并且周期数刚好满足需要
    if from_pd == to_pd and pd_number == len(data_list_processed):
        return data_list_processed
    # 转成period为1 也就是天，不够不零 ，超出截取
    days = to_pd * pd_number
    rd = []
    for d in data_list_processed:
        rd.extend([d / from_pd] * from_pd)
    # 截取days长度
    rd = rd[0:days]
    # 不足补零
    rd.extend([0] * (days - len(rd)))
    rt = [sum(rd[s: s + to_pd]) for s in range(0, len(rd), to_pd)]
    return rt


def day_to_period_step(to_period, period_number, data_list, step):
    """
    带步长的天到周期转换；剩余部分是一个周期，周期不够，补零；周期多，截取
    [1,2,3,4,5,6,7,8,9,10,11,12,13,14] 1-7 [28,36,44,52,60,68,76,84,92]
    step1 : [28.0, 35.0, 42.0, 49.0, 56.0, 63.0, 70.0, 77.0]
    step2 : [28.0, 42.0, 56.0, 70.0, 69.0]
    :param to_period:   转成什么单位
    :param predict_list: 要转换的数据[]
    :return:
    """
    # data_list = [float(d) for d in data_list]
    # data_list  可能有 'nan' [[u'1996321', u'682', u'nan', u'nan', u'nan', u'nan', u'-0.5', u'0.0']]
    data_list_processed = []
    for e in data_list:
        f = float(e)
        if math.isnan(f) or f is None:
            f = 0
        data_list_processed.append(f)
    rd = []
    if 1 == to_period:
        rd = data_list_processed
    else:
        # 数据如果不全，剩余部分凑最后一个周期
        if len(data_list) >= to_period:
            rd = [sum(data_list_processed[s: s + to_period]) for s in range(0, len(data_list) - to_period + 1, step)]
            if s + to_period < len(data_list) and s + step < len(data_list):
                rd.append(sum(data_list_processed[s + step:]))
        else:
            rd.append(sum(data_list))
        # 剩余补全
        rd.extend([0] * (period_number - len(rd)))
    return rd[0:period_number]


def day_to_period_mean_step(to_period, period_number, data_list, step):
    """
    不是求周期内多天的和，而是求多天的均值
    带步长的天到周期转换；剩余部分是一个周期，周期不够，补零；周期多，截取
    to_period =7
    period_number =2
    data_list =[1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    result =[4.0, 11.0]
    :param to_period:   转成什么单位
    :param predict_list: 要转换的数据[]
    :return:
    """
    # data_list = [float(d) for d in data_list]
    # data_list  可能有 'nan' [[u'1996321', u'682', u'nan', u'nan', u'nan', u'nan', u'-0.5', u'0.0']]
    data_list_processed = []
    # 缺失值 None 补全成四种方式

    for e in data_list:
        f = convert_to_float(e)
        data_list_processed.append(f)
    rd = []
    if 1 == to_period:
        rd = data_list_processed
    else:
        # 数据如果不全，剩余部分凑最后一个周期
        if len(data_list) >= to_period:
            rd = [mean(data_list_processed[s: s + to_period]) for s in range(0, len(data_list) - to_period + 1, step)]
            if s + to_period < len(data_list) and s + step < len(data_list):
                rd.append(mean(data_list_processed[s + step:]))
        else:
            rd.append(mean(data_list))
        # 剩余补全
        rd.extend([0] * (period_number - len(rd)))
    return rd[0:period_number]


def gen_key(line, all_key_map, group_by):
    """
    从line中取出 group_by中指定的key 部分
    :param line:
    :param all_key_map:
    :param group_by:
    :return:
    """
    return [line[all_key_map[e]] for e in group_by]


def gen_values(line, all_key_map, group_by):
    """
    从line中取出 不是group_by中指定的key部分（其他部分)
    :param line:
    :param all_key_map:
    :param group_by:
    :return:
    """
    not_in_group_by = list(set(all_key_map.keys()) ^ group_by)
    return [line[all_key_map[e]] for e in not_in_group_by]


def fill_with_near(values):
    """
    把一个包含None的数组用邻近值填充 (None,float('nan'),None,2,5,None,8,None,None) =>(2, 2, 2, 2, 5, 5, 8, 8, 8)
    会拷贝values
    :param values:
    :return:
    """
    pre = -1
    values = list(values)
    filled = False
    for i, v in enumerate(values):
        if v is not None and not math.isnan(v):
            if not filled:
                values[0:i] = [v] * i
                filled = True
            pre = i
        else:
            values[i] = values[pre]
    return values


def check_hadoop_file_exist(path):
    """
    检查hadoop中指定路径是否存在
    :param path:
    :return:
    """
    rs = os.system("hadoop fs -test -e " + path)
    return rs == 0


def gen_date_range_order_dict(start_day, period_number, step=1):
    """
    生成起始日期 指定天数的 日期字符串OrderedDict 返回 起始日期是start_day，步长为step的 period_number个日期字典
    start_date = dt.datetime.strptime('2016-08-01', "%Y-%m-%d").date()
    period_number =10
    step =2
    OrderedDict([('2016-08-01', None), ('2016-08-03', None), ('2016-08-05', None), ('2016-08-07', None), ('2016-08-09', None), ('2016-08-11', None), ('2016-08-13',
    None), ('2016-08-15', None), ('2016-08-17', None), ('2016-08-19', None)])
    :param start_day:
    :param period_number:
    :return:
    """
    date_list = [(start_day + dt.timedelta(d)).strftime('%Y-%m-%d') for d in range(0, period_number * step, step)]
    none_list = [None] * period_number
    return OrderedDict(zip(date_list, none_list))


def save_text_rdd(hdfs_path, rdd):
    """
    删除原来的hdfs_path，保存Rdd
    :param hdfs_path:
    :param rdd:
    :return:
    """
    os.system("hadoop fs -rm -r " + hdfs_path)
    rdd.saveAsTextFile(hdfs_path)
    return rdd


def save_pickle_rdd(hdfs_path, rdd):
    """
    删除原来的hdfs_path，保存Rdd
    :param hdfs_path:
    :param rdd:
    :return:
    """
    os.system("hadoop fs -rm -r " + hdfs_path)
    rdd.saveAsPickleFile(hdfs_path)
    return rdd


def init_table(table):
    """
    返回字段与序号对应关系
    :param table:
    :return:
    """
    return dict(zip(table, range(len(table))))


def rank(data_list, reverse=False):
    """
    每个元素的排行  有并列 现象 例如 1 2 2  -->1 2 2
    :param data_list:[4,7,9,10,6,11,3])
    :return:[1, 3, 4, 5, 2, 6, 0] +1
    """
    seq = sorted(data_list, reverse=reverse)
    return [seq.index(v) + 1 for v in data_list]


def writelines(file_path, metrics):
    """
    把metircs 内容输出到filepath文件中
    :param file_path:
    :param metrics:
    :return:
    """
    try:
        fo = open(file_path, "w+")
        fo.writelines(metrics)
    finally:
        fo.close()


def default_filter(data):
    """
    默认过滤器不加任何过滤
    :param data:
    :return:
    """
    return data


def get_dic_files(path):
    """
    获取目录下所有文件路径
    :param path:
    :return:
    """
    model_file_dates = os.popen('hadoop fs -ls ' + path).read()
    array = model_file_dates.split('\n')
    paths = []
    i = 0
    for ar in array:
        if 0 < i < len(array) - 1:
            words = ar.split(' ')
            paths.append(words[len(words) - 1])
        i += 1
    return paths


def check_error(file):
    """
    检查结果文件是否正常
    :param f:
    :return:
    """
    if isinstance(file, str):
        file = [file]
    for ff in file:
        if len(get_dic_files(ff)) < 2:
            return True
    return False


def convert_to_float(value):
    """
    把 value转成 float
    注意：'-100'.isdigit() False
    value：
    1.可能为None 返回 0
    2.不能转成数值的字符串 返回0
    3.数值 返回float()
    4.能够转成数值的字符串 返回 float()
    5.其他 返回 0


    :param value:
    :return:
    """
    # None 返回 注意：isdigit只能判断整数
    # if isinstance(value, numbers.Number):
    #     return float(value)
    # elif isinstance(value, six.string_types):
    #     tmp = value.replace('.', '', 1)
    #     tmp = tmp.replace('-', '', 1)
    #     if tmp.isdigit():
    #         return float(value)
    #     else:
    #         return 0
    # else:
    #     return 0
    v = 0
    try:
        v = float(value)
    except Exception:
        v = 0
    if math.isnan(v):
        v = 0
    return v


def convert_to_int(value):
    """
    把 value转成 float
    value：
    1.可能为None 返回 0
    2.不能转成数值的字符串 返回0
    3.数值 返回float()
    4.能够转成数值的字符串 返回 float()
    5.其他 返回 0

    :param value:
    :return:
    """
    # None 返回 注意：isdigit只能判断整数
    # if isinstance(value, numbers.Number):
    #     return int(value)
    # elif isinstance(value, six.string_types):
    #     tmp = value.replace('.', '', 1)
    #     tmp = tmp.replace('-', '', 1)
    #     if tmp.isdigit():
    #         return int(float(value))
    #     else:
    #         return 0
    # else:
    #     return 0
    v = 0
    try:
        v = int(float(value))
    except Exception:
        v = 0
    return v


def fill_month_measures(month_data, data_dates, miss_handle=1):
    """
    传入每天销量数组，求出指定周期长度的销量
    1. 'nan',None异常值处理 ，跟缺失值一样处理
    2. miss_handle:缺失值处理
        1>有任何一天销量缺失返回 None
        2>对缺失销量补 0
        3>对缺失销量补 mean
        4>对缺失销量使用临近值填充
    例子:
    支持直接年月方式 2016-01 2016-02
    month_data=[('2016-01-01',[1,1,1]),('2016-02-02',[2,2,2]),('2016-03-03',[3,3,3]),('2016-04-04',[4,4,4]),('2016-05-05',[5,5,5]),('2016-06-06',[6,6,6]),('2016-08-08',[8,8,8])]
    data_dates=['2016-01-01','2016-02-02','2016-03-03','2016-04-04','2016-05-05','2016-06-06','2016-07-07','2016-08-08']
    返回：[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0, 8.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0, 8.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0, 8.0]]
    :param month_sales:
    :param data_month:
    :param miss_handle
    :return:
    """
    # 指标个数
    measures_num = len(month_data[0][1])
    # 年月 --2016-01 ;
    data_months = [p[0:7] for p in data_dates]
    # 构造日期为键的空有序map ，指标值初始化为-1
    dt_dict = OrderedDict(zip(data_months, [[-1] * measures_num] * len(data_months)))
    # 如果date_sales有时间段内没有的日期，自动忽略
    # 'nan' None按缺失处理
    for md in month_data:
        ym = md[0][0:7]
        if dt_dict.has_key(ym):
            dt_dict[ym] = [convert_to_float(m) if m is not None and not math.isnan(float(m)) else -1 for m in md[1]]

    # 方法1：map(list,zip(*dt_dict.values())) map转换类型 map(list,[(1,2,3)]) --[[1, 2, 3]]
    # 方法2：[[row[i] for row in dt_dict.values()] for i in range(len(dt_dict.values()[0]))] --也可以 两重循环
    processed_data = map(list, zip(*dt_dict.values()))
    # 缺失值
    not_exist_num = [s for s in list(chain(*processed_data)) if s == -1]
    # 有缺失值返回 None
    if miss_handle == 1 and len(not_exist_num) > 0:
        return None
    # 有缺失值补零 注意：如果v ==-1 and 0 or v 就会永远执行 v **
    if miss_handle == 2:
        measures = [[v != -1 and v or 0 for v in m] for m in processed_data]
    elif miss_handle == 3:
        # 有缺失值补充 均值
        non_empty = [[v for v in m if v != -1] for m in processed_data]
        default_values = [mean(ne) for ne in non_empty]
        # v == -1是用均值替换
        # 注意：v == -1 and default_value  有bug，如果均值未0 会 执行 v 弄成-1
        # v != -1 and v or default_value   有bug :v == 0 时会用 default_value替换
        measures = []
        for i, m in enumerate(processed_data):
            ma = []
            for m1 in m:
                if m1 == -1:
                    ma.append(default_values[i])
                else:
                    ma.append(m1)
            measures.append(ma)
    else:
        # 按就近填充值
        pres = [-1] * measures_num
        filled_begins = [False] * measures_num
        measures = processed_data
        for i, measure in enumerate(measures):
            for j, v in enumerate(measure):
                if v != -1:
                    if not filled_begins[i]:
                        measure[0:j] = [v] * j
                        filled_begins[i] = True
                    pres[i] = j
                else:
                    measure[j] = measure[pres[i]]
    return measures


def agg_days_measures(day_data, data_dates, miss_handle=1):
    """
    传入每天销量数组，聚合指标
    1. 'nan',None异常值处理 ，跟缺失值一样处理
    2. miss_handle:缺失值处理
        1>有任何一天销量缺失返回 None
        2>对缺失销量补 0
        3>对缺失销量补 mean
        4>对缺失销量使用临近值填充
    例子:
    month_data=[('2016-01-01',[1,1,1]),('2016-02-02',[2,2,2]),('2016-03-03',[3,3,3]),('2016-04-04',[4,4,4]),('2016-05-05',[5,5,5]),('2016-06-06',[6,6,6]),('2016-08-08',[8,8,8])]
    data_dates=['2016-01-01','2016-02-02','2016-03-03','2016-04-04','2016-05-05','2016-06-06','2016-07-07','2016-08-08']
    :param month_sales:
    :param data_month:
    :param miss_handle
    :return:
    """
    # 指标个数
    measures_num = len(day_data[0][1])
    # 构造日期为键的空有序map ，指标值初始化为-1
    dt_dict = OrderedDict(zip(data_dates, [[-1] * measures_num] * len(data_dates)))
    # 如果date_sales有时间段内没有的日期，自动忽略
    # 'nan' None按缺失处理
    for dd in day_data:
        ymd = dd[0]
        if dt_dict.has_key(ymd):
            dt_dict[ymd] = [convert_to_float(d) if d is not None and not math.isnan(float(d)) else -1 for d in dd[1]]

    # 方法1：map(list,zip(*dt_dict.values())) map转换类型 map(list,[(1,2,3)]) --[[1, 2, 3]]
    # 方法2：[[row[i] for row in dt_dict.values()] for i in range(len(dt_dict.values()[0]))] --也可以 两重循环
    processed_data = map(list, zip(*dt_dict.values()))
    # 缺失值
    not_exist_num = [s for s in list(chain(*processed_data)) if s == -1]
    # 有缺失值返回 None
    if miss_handle == 1 and len(not_exist_num) > 0:
        return None
    # 有缺失值补零 注意：如果v ==-1 and 0 or v 就会永远执行 v **
    if miss_handle == 2:
        measures = [[v != -1 and v or 0 for v in m] for m in processed_data]
    elif miss_handle == 3:
        # 有缺失值补充 均值
        non_empty = [[v for v in m if v != -1] for m in processed_data]
        default_values = [mean(ne) for ne in non_empty]
        # v == -1是用均值替换
        # 注意：v == -1 and default_value  有bug，如果均值未0 会 执行 v 弄成-1
        # v != -1 and v or default_value   有bug :v == 0 时会用 default_value替换
        measures = []
        for i, m in enumerate(processed_data):
            ma = []
            for m1 in m:
                if m1 == -1:
                    ma.append(default_values[i])
                else:
                    ma.append(m1)
            measures.append(ma)
    else:
        # 按就近填充值
        pres = [-1] * measures_num
        filled_begins = [False] * measures_num
        measures = processed_data
        for i, measure in enumerate(measures):
            for j, v in enumerate(measure):
                if v != -1:
                    if not filled_begins[i]:
                        measure[0:j] = [v] * j
                        filled_begins[i] = True
                    pres[i] = j
                else:
                    measure[j] = measure[pres[i]]
    agg = [sum(m) for m in measures]
    return agg
