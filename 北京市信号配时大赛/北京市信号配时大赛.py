# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:48:13 2020
信号配时大赛
@author: 18120900
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn import metrics
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


def dealtrackdata():
    rootdir_open = r'.\三里河东路5月30日GPS坐标信息'
    filenames = os.listdir(rootdir_open)
    data = []
    for filename in filenames:
        path_open = os.path.join(rootdir_open, filename)
        if not os.path.isfile(path_open):
            continue
        temp_data = pd.read_csv(path_open, encoding='utf-8')
        temp_data['description'] = filename
        data.extend(temp_data.values)
    df = pd.DataFrame(data=data)
    with pd.ExcelWriter('result.xlsx') as writer:
        df.to_excel(writer)


def combainexcels():
    """
    合并多个excel表
    @return: excel表
    """
    rootdir_open = r"F:\18120900\桌面\北京交通信号配时大赛\原始数据\五棵松周边数据\路段数据-四环西"
    filenames = os.listdir(rootdir_open)
    result = pd.DataFrame()
    for filename in filenames:
        path_open = os.path.join(rootdir_open, filename)
        if not os.path.isfile(path_open):
            continue
        if result.empty:
            result = pd.read_excel(path_open, 0)
        else:
            result = result.append(pd.read_excel(path_open, 0))
        print(filename, result.size)
    result.to_excel(os.path.join(rootdir_open, '路段数据汇总表-四环西.xlsx'))


def plotsegmentdata():  # 画路段速度变化曲线图
    sheets = pd.read_excel('路段数据汇总表.xlsx', 1)
    sheets['日期'] = sheets['日期'].apply(lambda x: datetime.datetime.strftime(x, '%m-%d'))
    sheets['时间'] = sheets['时间'].apply(lambda x: datetime.time.strftime(x, '%H:%m'))

    # 画出所有路段的速度变化曲线
    fig, axis = plt.subplots(figsize=(5.5, 3), nrows=2, ncols=1, sharex='col')
    plt.subplots_adjust(0.09, 0.15, 0.8, 0.93, 0.2, 0.01)
    sns.lineplot(x='时间', y='速度', hue='路段', style='方向', data=sheets, ax=axis[0], legend=False)
    sns.lineplot(x='时间', y='拥堵长度', hue='路段', style='方向', data=sheets, ax=axis[1])
    plt.suptitle('各路段速度变化曲线')
    plt.xlabel(xlabel='')
    plt.xticks(rotation=90)
    plt.legend(labelspacing=0.2, frameon=False, bbox_to_anchor=(1, 1), loc=6)
    plt.show()

    # 画出每条路段的速度变化曲线
    for name, group in sheets.groupby(by='路段'):
        fig, axis = plt.subplots(figsize=(5.5, 3), nrows=2, ncols=1, sharex='col')
        plt.subplots_adjust(0.09, 0.15, 0.84, 0.93, 0.2, 0.01)
        sns.lineplot(x='时间', y='速度', hue='日期', style='方向', ci=0, data=group, ax=axis[0], legend=False)
        sns.lineplot(x='时间', y='拥堵长度', hue='日期', style='方向', ci=0, data=group, ax=axis[1])
        plt.suptitle(name)
        plt.xlabel(xlabel='')
        plt.xticks(rotation=90)
        plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
        plt.legend(labelspacing=0.2, frameon=False, bbox_to_anchor=(1, 1), loc=6)
        plt.show()


def plottrackspacetimediagram():  # 画车辆轨迹时空图
    # plt.figure(figsize=(4, 3.5))
    # data = pd.read_excel('轨迹数据.xlsx', 0)
    # data['distance'] = data.apply()
    # for name, group in data.groupby(by='description'):
    #     time, distance = [0], [0]
    #     for row in group.sort_values(by=['time'], ascending=True).values.tolist():
    #         time.append(len(time))
    #         distance.append()
    #     plt.plot(time, distance, 'go-')
    # plt.tight_layout()
    # plt.show()
    pass


def drawcartrack(sheet_index):  # 画车辆轨迹图
    """
    @param sheet_index: [0] data;[1] crossroadinfo
    @return: None
    """
    plt.figure(figsize=(4, 3.5))
    plt.subplots_adjust(0.2, 0.05, 0.98, 0.98, 0.2, 0.2)
    data = pd.read_excel('轨迹数据.xlsx', sheet_index)
    track, crossroad = data[sheet_index[0]], data[sheet_index[1]]
    # 画出车辆的行驶轨迹，每条轨迹的经度调高i从而使轨迹分开
    grouped_data, i = track.groupby(['description']), 0
    for name, group in grouped_data:
        plt.scatter(group.gdlongitude + i, group.gdlatitude, label=name, s=1)
        i += 0.0003

    # 画出交叉口所在位置
    x = plt.xlim()
    for i in range(len(crossroad)):
        plt.plot(x,
                 [crossroad.loc[i, 'gdlatitude'],
                  crossroad.loc[i, 'gdlatitude']],
                 'k-',
                 linewidth=1)

    # 各种图表参数设置
    plt.yticks(crossroad.gdlatitude.tolist(), crossroad.name.tolist())
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
    plt.legend()


def dealflowtable():
    """
    用来处理三里河东路交通量调查表
    @return:
    """
    rootdir_open = r'.\2020年5月30日（周六）路口数据'
    filenames = os.listdir(rootdir_open)
    index_dict = {'路口机动车流量调查表': 0, '路口非机动车流量调查表': 1,
                  '路口行人流量调查表': 2, '行人过街流量调查表': 3}
    title = ['cname', 'ctype', 'cdate', 'capproach', 'time']
    result = [[title], [title], [title], [title]]
    for filename in filenames:
        path_open = os.path.join(rootdir_open, filename)
        if not os.path.isfile(path_open):
            continue
        sheets = pd.read_excel(path_open, sheet_name=None)  # 读取工作簿中所有excel表
        for sheet in sheets.items():
            tabel_index = index_dict[sheet[1].columns[0].strip()]
            cname = sheet[1].iloc[0, 2]
            ctype = sheet[1].iloc[1, 2]
            cdate = sheet[1].iloc[2, 2]
            capproach = sheet[1].iloc[3, 12] if tabel_index == 0 else ''

            temp_result = {
                0: lambda x: [x.iloc[4:46, 0:15]],
                1: lambda x: [x.iloc[4:46, 0:9]],
                2: lambda x: [x.iloc[4:46, 0:13]],
                3: lambda x: [x.iloc[4:46, 0:6]]
            }

            sheet = sheet[1].apply(lambda x: x.replace('/', 0))
            tr = temp_result[tabel_index](sheet)[0].values
            temp_time = ''
            for i in range(tr.shape[0]):
                if tr[i, 0] != '' and isinstance(tr[i, 1], float) and np.isnan(tr[i, 1]):
                    temp_time = tr[i, 0]
                    continue
                startcontent = [cname, ctype, cdate, capproach, temp_time]
                startcontent.extend(tr[i, :])
                result[tabel_index].append(startcontent)

    with pd.ExcelWriter('流量调研数据处理结果.xlsx') as writer:
        for key, value in index_dict.items():
            df = pd.DataFrame(data=result[value])
            df.to_excel(writer, index=False, sheet_name=key)


def kmeans(sheet, n_clusters, looptimes=10):
    """
    K_Means聚类分析模型
    @param sheet: 设局集
    @param n_clusters: 聚类中心数
    @param looptimes: 聚类次数
    @return: None
    """
    x = sheet.values[1:]
    elbowmethod(x)
    silhouettecoefficient(x)
    result = []
    for i in range(looptimes):
        result.append(KMeans(n_clusters=n_clusters).fit(x).predict(x).tolist())
    print(result)


def elbowmethod(data):
    """
    “肘”方法：核心指标是SSE（sum of the squared errors,误差平方和，
    即所有样本的聚类误差（累计每个簇中样本到质心距离的平方和），随着K的增大每个簇
    聚合度会增强，SSE下降幅度会增大，随着K值继续增大SSE的下降幅度会减少并趋于平缓
    SSE和K值的关系图会呈现一个手肘的形状，此时肘部对应的k值就是最佳的聚类数。
    """
    K = range(1, 9)  # 假设可能聚类成1-8类
    lst = []
    for k in K:
        kmeans_model = KMeans(n_clusters=k).fit(data)
        # 计算对应k值时最小值列表和的平均值
        # cdist(data, kmeans.cluster_centers_, 'euclidean')求data到各质心
        # cluster_centers_之间的距离平方和，'euclidean'表示使用欧式距离计算
        dist = cdist(data, kmeans_model.cluster_centers_, 'euclidean')
        lst.append(sum(np.min(dist, axis=1)) / data.shape[0])
    plt.figure(figsize=(5, 4))
    plt.subplots_adjust(0.13, 0.12, 0.98, 0.93, 0.2, 0.2)
    plt.plot(K, lst, 'bo-')
    plt.title('Elbow method')
    plt.xlabel('K')
    plt.ylabel('Cost function')
    plt.show()


def silhouettecoefficient(data):
    """
    轮廓系数法：结合聚类的凝聚度（Cohesion）和分离度（Seperation）来考虑，凝聚度为
    样本与同簇其他样本的平均距离，分离度为样本与最近簇中所有样本的平均距离，该值处于-1到1
    之间，值越大表示聚类效果越好。
    """
    K = range(2, 9)  # 假设可能聚成2-8类
    lst = []
    for k in K:
        kmeans_model = KMeans(n_clusters=k).fit(data)
        # silhouette_score()计算所有样本的平均轮廓系数
        # kmeans_model.labels_ 每个样本预测的类标签
        # metric='euclidean' 使用欧式距离计算
        sc_score = metrics.silhouette_score(data, kmeans_model.labels_, metric='euclidean')
        lst.append(sc_score)
    plt.figure(figsize=(5, 4))
    plt.subplots_adjust(0.13, 0.12, 0.98, 0.93, 0.2, 0.2)
    plt.plot(K, lst, 'bo-')
    plt.title('Silhouette Coefficient')
    plt.xlabel('K')
    plt.ylabel('Score')
    plt.show()


def divisionoftimeperiod(data):
    """
    时段划分模型
    @param data: 数据集
    @return: None
    """
    title = data.columns.values.tolist()
    result = []
    # 生成21*21的矩阵(7天，每天三个纬度)，即一天与其他天的余弦值
    for c1 in range(len(title)):
        temp = []
        for c2 in range(len(title)):
            temp.append(cosine_similarity(data.iloc[:, c1], data.iloc[:, c2]))
        result.append(temp)
    df = pd.DataFrame(columns=title, data=result, index=title)
    plt.figure(figsize=(5, 4))
    plt.subplots_adjust(0.09, 0.1, 0.98, 0.98, 0.2, 0.2)
    sns.heatmap(df, cmap='coolwarm')
    plt.show()


def cosine_similarity(x, y, norm=False):
    """
    计算两个向量x和y的余弦相似度
    @param x: 相量X
    @param y: 相量Y
    @param norm: 结果是否归一化到【0，1】区间
    @return: double
    """
    assert len(x) == len(y), "len(x) != len(y)"
    if np.all(x == 0) or np.all(y == 0):
        return float(1) if x == y else float(0)

    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
    # 归一化到[0, 1]区间内；默认不归一化
    return 0.5 * cos + 0.5 if norm else cos


def subareadivision():
    sheets = pd.read_excel('子区划分.xlsx', [0, 1, 2])
    # 读取交叉口的周期长度
    cyclelengthes = sheets[0].set_index('code').T.to_dict(orient='list')
    default_c = np.zeros(len(sorted(cyclelengthes.values())[0]))
    # 读取相邻交叉口之间的距离
    distances = sheets[1].set_index(['code1', 'code2']).T.to_dict(orient='list')
    default_d = np.zeros(len(sorted(distances.values())[0]))
    # 读取每个交叉口的流量数据
    flows = sheets[2].set_index(['code', 'date', 'time']).T.to_dict(orient='list')
    default_f = np.zeros(len(sorted(flows.values())[0]))

    date = 20191202                # 选取哪一天的数据
    time = 18                      # 选取一天中哪个小时的数据
    threshold_r = 0.9              # 初始分离阈值，大于该值时划分为两个子区，迭代时该值不断递增
    threshold_q = 0.3              # 模块度阈值，大于该值时为最优划分结果
    q = -1                         # 模块度初始值
    a = 2.6                        # 常数，取2.6，用来计算关联度
    alpha1 = 0.06                  # 路段关联流量关联系数，用来计算关联度
    alpha2 = 0.002                 # 路段长度关联系数，用来计算关联度
    vc_min = 0.02                  # 基本通行能力乘以的系数
    vc_max = 0.75                  # 基本通行能力乘以的系数
    basic_traffic_capacity = 2000  # 基本通行能力
    R_Q_dict = {}                  # key=（code1，code2）,value=(r,q)，code1 < code2
    # distances中code2在code1的方向（NID等）在flows中对应的进口道流量的index
    direction_dict = {'NID': 0, 'EID': 1, 'SID': 2, 'WID': 3, 'NEID': 0, 'SEID': 1, 'SWID': 2, 'NWID': 3}

    # key=code;value=[小区编号，是否为独立小区, 度]
    key = set(cyclelengthes.keys())
    value = [[i + 1, False, 0] for i in range(len(key))]
    result_dict = dict(zip(key, value))

    # 计算每个交叉口的度，也即是每个交叉口与相邻交叉口之间的路段数
    temp_intersectiones = [key[0] for key in distances.keys()]
    for k, v in result_dict.items():
        v[2] = temp_intersectiones.count(k)

    # 基于向外扩张的思想，生成交通控制子区初始划分结果
    for key, value in distances.items():
        direction = value[0]
        distance = value[1]
        # 是否满足周期划分条件
        conditions1 = abs(cyclelengthes.get(key[0], default_c)[0] - cyclelengthes.get(key[1], default_c)[0]) <= 10
        # 是否满足流量划分条件
        vc = flows.get((key[0], date, time), default_f)[direction_dict[direction]] / basic_traffic_capacity
        conditions2 = (vc_min < vc < vc_max)
        # 是否满足路段长度划分条件
        conditions3 = distance <= 700
        # 如果同时满足以上三个条件，则把交叉口放在同一个子区，否则划分为不同子区
        if conditions1 and conditions2 and conditions3:
            # 判断交叉口是否在字典里面，若不在则添加到字典里，子区编号设为key的数量+1
            if key[0] not in result_dict:
                result_dict[key[0]] = [len(result_dict.keys()) + 1, False, result_dict[key[0]][2]]
            if key[1] not in result_dict:
                result_dict[key[1]] = [len(result_dict.keys()) + 1, False, result_dict[key[0]][2]]

            # 判断两个交叉口中某一个是否已经划分小区了
            if result_dict[key[1]][1] or result_dict[key[0]][1]:
                if result_dict[key[0]][1]:  # 若code1交叉口已经划分小区
                    result_dict[key[1]] = [result_dict[key[0]][0], True, result_dict[key[0]][2]]
                else:
                    result_dict[key[0]] = [result_dict[key[1]][0], True, result_dict[key[0]][2]]
            else:  # 若code1和code2交叉口都未划分小区
                result_dict[key[1]] = [result_dict[key[0]][0], True, result_dict[key[0]][2]]
                result_dict[key[0]][1] = True

    # 计算所有交叉口之间的关联度
    for i in result_dict.keys():
        for j in result_dict.keys():
            key = (j, i) if i > j else (i, j)
            v = distances.get((i, j), default_d)
            if v[0] in direction_dict.keys():
                temp_flow = flows.get((i, date, time), default_f)[direction_dict[v[0]]]
            else:
                temp_flow = 0
            R_Q_dict[key] = [1 / (1 + abs(a - alpha1 * temp_flow + alpha2 * v[1])), 0]

    while q < threshold_q:
        threshold_r += 0.01
        # 根据交叉口关联度对初始划分方案进行调整
        for k in R_Q_dict.keys():
            isinonearea = (result_dict[k[0]][0] == result_dict[k[1]][0])  # 是否在一个小区
            isgreater = (R_Q_dict[k][0] >= threshold_r)   # 关联度是否大于分离阈值时
            if isinonearea and isgreater:
                result_dict[k[1]][0] = max([v[0] for v in result_dict.values()]) + 1

        # 计算小区的组合关联度
        area_dict = {}
        for key in result_dict.keys():
            if result_dict[key][0] not in area_dict:
                area_dict[result_dict[key][0]] = [key]
            else:
                area_dict[result_dict[key][0]].append(key)

        # 比较小区组合关联度ra和子区边界关联度rax
        for k in result_dict.keys():
            for key, value in area_dict.items():
                if calarear(value, k, R_Q_dict):
                    result_dict[k][0] = key

        # 计算划分结果的模块度
        R_Q_dict = calq(result_dict, distances, R_Q_dict)
        q = sum([v[1] for v in R_Q_dict.values()]) / len(distances)
        print('模块度：' + str(q))

    df = pd.DataFrame.from_dict(result_dict, orient='index')
    with pd.ExcelWriter('子区划分结果.xlsx') as writer:
        df.to_excel(writer)


def calarear(intersectiones, intersection, r_q_dict):
    tuple_interes = [(i, j) for i in intersectiones for j in intersectiones]
    ra = sum([r_q_dict.get(k, [0, 0])[0] for k in tuple_interes])

    rax = 0
    for i in intersectiones:
        intersection, i = (i, intersection) if i < intersection else (intersection, i)
        rax += r_q_dict.get((intersection, i), [0, 0])[0]
    # 需要有等于号，因为存在两个独立且不相连的交叉口
    return ra < rax


def calq(result_dict, distances, r_q_dict):
    m = len(distances)  # 子区中边的个数*2
    for key in r_q_dict.keys():
        if result_dict[key[0]][0] != result_dict[key[1]][0]:  # 如果两个交叉口不在一个小区
            r_q_dict[key][1] = 0
            continue
        if (key[0], key[1]) in distances:  # 如果两个交叉口直接相连
            r_q_dict[key][1] = (1 - result_dict[key[0]][2] * result_dict[key[1]][2] / m)
        else:  # 如果两个交叉口不直接相连
            r_q_dict[key][1] = (0 - result_dict[key[0]][2] * result_dict[key[1]][2] / m)
    return r_q_dict


def extra_los_delay():
    rootdir_open = r"F:\18120900\桌面\北京交通信号配时大赛\交通信号配时大赛-决赛\Synchro模型\0-文件报告"
    filenames = os.listdir(rootdir_open)
    result = []
    for filename in filenames:
        path_open = os.path.join(rootdir_open, filename)
        if not os.path.isfile(path_open):
            continue
        print(filename)
        with open(path_open, 'r', encoding='ANSI', errors='ignore') as f:
            for line in f.readlines():
                line = line.strip()
                if line[len(line)-9::] == '2020/9/22':
                    intersectionname = line[0:len(line)-9]
                    result.append(filename + '\t' + intersectionname + '\t' + line)
                if line[0:12] in ['Volume (vph)  ', 'Approach Del', 'Approach LOS', 'Lane Group  ']:
                    result.append(filename + '\t' + intersectionname + '\t' + line)
    f = open(os.path.join(rootdir_open, 'result2.txt'), 'w')
    f.write('\n'.join(result))
    f.close()


def plot_los_delay():
    filepath = r"F:\18120900\桌面\北京交通信号配时大赛\交通信号配时大赛-决赛\Synchro模型\0-文件报告\result2.csv"
    data = pd.read_csv(filepath, encoding='gb2312')
    for name, group in data.groupby('intersection'):
        width = 0.3  # the width of the bars
        fig, axis = plt.subplots(figsize=(5.5, 4), nrows=3, ncols=1, sharex='col')
        plt.subplots_adjust(0.1, 0.1, 0.98, 0.88, 0.2, 0.2)

        data1 = group.loc[lambda d: d['period'] == '9.8早']
        x = np.array(data1.loc[data1['isoptimized'] == '优化前', 'direction'])
        label11 = data1.loc[data1['isoptimized'] == '优化前'].set_index('direction').to_dict()['los']
        rects11 = axis[0].bar(x - width / 2, data1.loc[data1['isoptimized'] == '优化前', 'delay'], width, label='优化前')
        label12 = data1.loc[data1['isoptimized'] == '优化后'].set_index('direction').to_dict()['los']
        rects12 = axis[0].bar(x + width / 2, data1.loc[data1['isoptimized'] == '优化后', 'delay'], width, label='优化后')
        axis[0].set_ylabel('延误/s')
        axis[0].set_xlabel('早高峰')

        data2 = group.loc[lambda d: d['period'] == '9.8午']
        x = np.array(data2.loc[data2['isoptimized'] == '优化前', 'direction'])
        label21 = data2.loc[data2['isoptimized'] == '优化前'].set_index('direction').to_dict()['los']
        rects21 = axis[1].bar(x - width / 2, data2.loc[data2['isoptimized'] == '优化前', 'delay'], width, label='优化前')
        label22 = data2.loc[data2['isoptimized'] == '优化后'].set_index('direction').to_dict()['los']
        rects22 = axis[1].bar(x + width / 2, data2.loc[data2['isoptimized'] == '优化后', 'delay'], width, label='优化后')
        axis[1].set_ylabel('延误/s')
        axis[1].set_xlabel('午高峰')

        data3 = group.loc[lambda d: d['period'] == '9.8晚']
        x = np.array(data3.loc[data3['isoptimized'] == '优化前', 'direction'])
        label31 = data3.loc[data3['isoptimized'] == '优化前'].set_index('direction').to_dict()['los']
        rects31 = axis[2].bar(x - width / 2, data3.loc[data3['isoptimized'] == '优化前', 'delay'], width, label='优化前')
        label32 = data3.loc[data3['isoptimized'] == '优化后'].set_index('direction').to_dict()['los']
        rects32 = axis[2].bar(x + width / 2, data3.loc[data3['isoptimized'] == '优化后', 'delay'], width, label='优化后')
        axis[2].set_ylabel('延误/s')
        axis[2].set_xlabel('晚高峰')
        plt.xticks([0, 1, 2, 3], ['西进口', '东进口', '南进口', '北进口'])

        autolabel(rects11, axis[0], label11)
        autolabel(rects12, axis[0], label12)
        autolabel(rects21, axis[1], label21)
        autolabel(rects22, axis[1], label22)
        autolabel(rects31, axis[2], label31)
        autolabel(rects32, axis[2], label32)
        plt.suptitle(name)
        plt.legend(bbox_to_anchor=(0.5, 3.3), loc=8, ncol=2, frameon=False)
        plt.show()


def autolabel(rects, ax, label):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{0}\n{1}'.format(height, label[int(rect.get_x() + 0.3)]),
                    xy=(rect.get_x() + rect.get_width() / 2, 0),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def main():
    sheets = pd.read_excel('Data.xlsx', [0, 1, 2])
    #
    # dealtrackdata()                              # 处理三里河东路车辆GPS轨迹数据
    # drawcartrack([0, 2])                         # 画车辆GPS轨迹数据
    # drawcartrack([1, 2])                         # 画车辆GPS轨迹数据
    # combainexcels()                              # 合并多个表格
    # plotsegmentdata()                            # 画路段数据图，2020.8.22-8.25
    # dealflowtable()                              # 汇总流量调查数据表
    # divisionoftimeperiod(sheets[0])              # 划分时段
    #
    # elbowmethod(sheets[1].values[1:])            # 画"肘"图来确定最好的聚类中心
    # silhouettecoefficient(sheets[1].values[1:])  # 画轮廓图来确定最好的聚类中心
    # kmeans(sheets[1], 6)                         # 工作日聚类分析结果
    # elbowmethod(sheets[2].values[1:])            # 画”肘“图来确定最好的聚类中心
    # silhouettecoefficient(sheets[2].values[1:])  # 画轮廓图来确定最好的聚类中心
    # kmeans(sheets[2], 6)                         # 休息日聚类分析结果
    #
    # subareadivision()                            # 交通子区划分
    # plt.show()


if __name__ == '__main__':
    # 由于变量作用域的问题，导致‘__main__’中的变量可能会覆盖子函数中的变量
    sns.set_style("darkgrid")
    plt.rcParams['font.sans-serif'] = ['simsun']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    # plt.rcParams['font.size'] = '12'  # 设置字体大小
    # main()
    # extra_los_delay()
    plot_los_delay()
    print('Done')
