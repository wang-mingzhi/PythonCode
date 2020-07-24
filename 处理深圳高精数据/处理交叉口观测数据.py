# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 20:25:29 2020
ProjectedName: 处理深圳交叉口车辆行驶轨迹高精数据
Software: PyCharm
author: 18120900
"""
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def dealdata():
    rootdir_open = r'F:\18120900\Documents\PythonCode\处理深圳高精数据\RawData'
    rootdir_save = r'F:\18120900\Documents\PythonCode\处理深圳高精数据'
    filenames = os.listdir(rootdir_open)
    for filename in filenames:
        path_open = os.path.join(rootdir_open, filename)
        path_save = os.path.join(rootdir_save, filename)
        if not os.path.isfile(path_open):
            continue

        contentList = []
        with open(path_open, encoding='utf-8') as f:
            data = f.readlines()
            json_file = [json.loads(item) for item in data]
            title = list(json_file[0].keys())[:-1]
            title.extend(list(json_file[0]['track'][0].keys()))
            for items in json_file:
                ct1 = [v for v in items.values() if not isinstance(v, list)]
                ct2 = [list(item.values()) for item in items['track']]
                for item in ct2:
                    ct = ct1.copy()
                    ct.extend(item)
                    contentList.append(ct)

        df = pd.DataFrame(columns=title, data=contentList)
        df.sort_values(by='time', ascending=True, inplace=True)
        df.drop(axis=0, columns=['track'], inplace=True)
        df['time2'] = df['time'].apply(lambda x: datetime.fromtimestamp(x / 1000.0).strftime("%Y-%m-%d %H:%M:%S.%f"))
        df.to_csv(path_save + '.csv', index=False, encoding='utf_8_sig')
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), filename, ' Done')


def isincoil(entryroad, n, e):
    lan, p = [], [entryroad, n, e]
    if p[0] in [4, 3, '4', '3']:  # 北向南
        lan = [[v['LaneType'], v['code']] for v in coils.values() if v['code'] in [4, 3, '4', '3'] and
               abs(p[1] - v['N1']) <= 8 and ((p[2] - v['E1']) * (p[2] - v['E2'])) <= 0]
    elif p[0] in [2, 1, '2', '1']:  # 西向东
        lan = [[v['LaneType'], v['code']] for v in coils.values() if v['code'] in [2, 1, '2', '1'] and
               abs(p[2] - v['E1']) <= 8 and ((p[1] - v['N1']) * (p[1] - v['N2'])) <= 0]
    return 0 if lan == [] else lan


def extractdata(data, result):
    index = pd.Series(range(len(data.columns.to_list())), index=data.columns.to_list())
    time_index, trackID_index, entryRoad_index = index['time'], index['trackID'], index['entryRoad']
    n_index, e_index = index['N'], index['E']

    ptime, btime = 0, 0
    pcars, bcars = {}, {}  # 记录前一时刻之前车辆是否经过线圈，key=(traceid,time)，value=True/False
    data = data.sort_values(by=['time'], ascending=True).values.tolist()
    for row in data:  # data.values比data.iterrows()快30多倍
        # 如果前一条数据的时刻和当前数据的时刻不相同，也即是时间发生了变化
        if ptime != row[time_index]:
            btime, bcars = ptime, pcars
            pcars = {}

        ptime, trackID = row[time_index], row[trackID_index]
        pcars[(trackID, ptime)] = bcars.get((trackID, btime), False)
        temp = 0 if pcars[(trackID, ptime)] else isincoil(row[entryRoad_index], row[n_index], row[e_index])
        if temp != 0:
            row.extend(temp[0])
            result.append(row)
            pcars[(trackID, ptime)] = True


def plot(coil):
    x, y = [], []
    for k, v in coil.items():
        x.extend(v[5::2])
        y.extend(v[6::2])
    plt.scatter(x, y)
    plt.show()


def statisticsdata(data):
    # 对结果进行统计
    data['count'] = 1
    data['time'] = pd.to_datetime(data['time'])
    with pd.ExcelWriter(statisticsresult) as writer:
        pivoted_data = data.pivot_table(values='count',
                                        index='time',
                                        columns=['entryRoad2', 'direction'],
                                        aggfunc=np.sum)
        dd = pivoted_data.resample('1T').sum()
        dd.to_excel(writer, sheet_name='间隔1分钟')
        dd = pivoted_data.resample('5T').sum()
        dd.to_excel(writer, sheet_name='间隔5分钟')

        pivoted_data2 = data.pivot_table(values='count',
                                         index='time',
                                         columns=['entryRoad2', 'direction', 'vehicleType'],
                                         aggfunc=np.sum)
        dd = pivoted_data2.resample('1T').sum()
        dd.to_excel(writer, sheet_name='间隔1分钟-分车型')
        dd = pivoted_data2.resample('5T').sum()
        dd.to_excel(writer, sheet_name='间隔5分钟-分车型')
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Done')


def main():
    filenames = os.listdir(rootdir_open)
    title, result = [], []
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Begin')
    for filename in filenames:
        path_open = os.path.join(rootdir_open, filename)
        data = pd.read_csv(path_open, encoding='utf-8')
        data = data[(data.N <= 2507550) & (data.N >= 2507200) & (data.E <= 814350) & (data.E >= 814100)]
        title = data.columns.to_list() if title == [] else title
        extractdata(data, result)
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), len(data), len(result), filename)

    title.extend(['direction', 'entryRoad2'])
    df = pd.DataFrame(data=result, columns=title)
    df['time'] = df['time'].apply(lambda x: datetime.fromtimestamp(x / 1000.0).strftime("%Y-%m-%d %H:%M:%S.%f"))
    df.to_excel(dealedresult, index=False)
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Finished')
    statisticsdata(pd.read_excel(dealedresult, 0))


if __name__ == "__main__":
    # 处理交叉口原始数据
    # dealdata()

    timelist = []
    coils = pd.read_csv("交叉口线圈设置.csv", encoding='utf-8-sig')
    coils = coils.set_index('LaneID').T.to_dict()
    # plot(coils)

    rootdir_open = r"F:\18120900\Documents\PythonCode\处理深圳高精数据\Data"
    statisticsresult = '交通量统计结果-按时段划分2.xlsx'
    dealedresult = '交通量统计结果-细分结果2.xlsx'
    main()
