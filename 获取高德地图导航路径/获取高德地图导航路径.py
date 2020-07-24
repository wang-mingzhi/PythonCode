# -*- coding: utf-8 -*- 
"""
Author: 18120900
Created: 2020/6/23 19:41
Software: PyCharm
Desc: 
"""
import pandas as pd
import numpy as np
import requests
import json
import time
import copy


def getraveltime(data, travelmodel, params):
    """
    解析json数据
    travelmodel =0驾驶路径规划;=1火车路径规划
    """
    title = ['起点', '终点']
    result = []

    for index, row in data.iterrows():
        if index % 50 == 0:
            print('共{1}个数据;已处理{0}个数据'.format(index, len(data)))
            time.sleep(1)

        params['origin'] = str(row['olon']) + ',' + str(row['olat'])
        params['destination'] = str(row['dlon']) + ',' + str(row['dlat'])

        if travelmodel == 0:
            js = crawler(travelmodel, params)
            if title == ['起点', '终点']:
                title.extend([k for k, v in js.items() if isinstance(v, str)])
                title.extend([k for k, v in js['route'].items() if isinstance(v, str)])
                title.extend([k for k, v in js['route']['paths'][0].items() if isinstance(v, str)])

            if not js['status']:
                print('Error in rows:{0}'.format(row))
                continue

            temp_result = [row['ocity'], row['dcity']]
            temp_result.extend([v for k, v in js.items() if isinstance(v, str)])
            temp_result.extend([v for k, v in js['route'].items() if isinstance(v, str)])
            temp = copy.deepcopy(temp_result)
            for j in range(int(js['count'])):
                temp_result = copy.deepcopy(temp)
                temp_result.extend([v for k, v in js['route']['paths'][j].items() if isinstance(v, str)])
                result.append(temp_result)
        else:
            params['city'] = row['oadcode']
            params['cityd'] = row['dadcode']
            js = crawler(travelmodel, params)
            title = ['起点', '终点', '方案数', 'origin', 'destination', '费用', '距离', '时间', '乘火车的里程', '乘火车的时间']
            temp = [row['ocity'], row['dcity'], js['count'], js['route']['origin'], js['route']['destination']]
            if js['count'] == 0:
                temp_result = copy.deepcopy(temp)
                temp_result.extend([0, 0, 0, 0, 0])
                result.append(temp_result)
            else:
                for j in range(int(js['count'])):
                    transit = js['route']['transits'][j]
                    temp_result = copy.deepcopy(temp)
                    temp_result.extend([transit['cost'], transit['distance'], transit['duration']])
                    dt = [[v['railway'].get('distance', 0), v['railway'].get('time', 0)] for v in transit['segments']]
                    dt = np.array(dt).astype(float)
                    temp_result.extend(dt.sum(axis=0))
                    result.append(temp_result)

    return result, title


def crawler(travelmodel, params):

    if travelmodel == 0:
        url = 'https://restapi.amap.com/v3/direction/driving?'
    elif travelmodel == 1:
        url = 'https://restapi.amap.com/v3/direction/transit/integrated?'
    else:
        url = 'https://restapi.amap.com/v3/direction/driving?'

    requests.DEFAULT_RETRIES = 5
    res = requests.get(url, params, timeout=25)
    res.raise_for_status()  # 如果响应状态码不是 200，就主动抛出异常
    js = json.loads(res.text)
    res.close()
    return js


if __name__ == '__main__':

    m_sheet = pd.read_excel('经纬度坐标.xlsx', sheet_name=0)

    # key为从高德地图网站申请的秘钥
    # 驾驶出行：strategy=0速度优先；=1费用优先；=2距离优先；=3速度优先；=4躲避拥堵；=5同时使用速度优先、费用优先、距离优先
    # 公共交通：strategy=0最快捷；=1最经济；=2最少换乘；=3最少步行；=5不乘地铁
    # origin出发点，这里不用填写
    # destination目的地，这里不用填写
    m_params = {'key': '填写自己的api密钥',
                'origin': '',
                'destination': '',
                'strategy': 0}

    # travelmodel =0驾驶路径规划;=1火车路径规划
    m_result, m_title = getraveltime(m_sheet, travelmodel=1, params=m_params)
    df = pd.DataFrame(m_result, columns=m_title, index=None)
    df.to_excel('路径规划结果.xlsx')
    print('Done')
