# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 10:40:59 2020
project name:读取高德地图里面的收费站数据
project name:把高德地图坐标系下的经纬度转为WGS84坐标系下的经纬度
@author: 18120900
"""

import requests
import json
import pandas as pd
import math
import time


def getstationinfo(url, params):
    """
    @param url: str
    @param params: dict
    @return: Dataframe
    """
    js = crawler(url, params)
    name = [key for key, value in js['pois'][0].items() if isinstance(value, str)]
    pagecount = -(-int(js['count']) // params['offset'])
    print('total station count:', js['count'])
    temp_list = []
    for i in range(pagecount):
        params['page'] = i + 1
        js = crawler(url, params)
        if js['status'] == '1':
            for v in js['pois']:
                temp_list.append([value for value in v.values() if isinstance(value, str)])
        else:
            print('meet an error in page:', params['page'])
            continue
        print('processing:', (params['page'] - 1) * params['offset'] + len(js['pois']))
    return pd.DataFrame(columns=name, data=temp_list)


def crawler(url, params):
    requests.DEFAULT_RETRIES = 5
    res = requests.get(url, params, timeout=25)
    res.raise_for_status()  # 如果响应状态码不是 200，就主动抛出异常
    js = json.loads(res.text)
    res.close()
    time.sleep(5)
    return js


def GCJ2WGS(location):
    """
    @param location: locations[1] = "113.923745,22.530824"
    @return: str: wgsLon,wgsLat
    """
    # 官方API: http://lbs.amap.com/api/webservice/guide/api/convert
    # 坐标体系说明：http://lbs.amap.com/faq/top/coordinate/3
    # GCJ02->WGS84 Java版本：http://www.cnblogs.com/xinghuangroup/p/5787306.html
    # 验证坐标转换正确性的地址：http://www.gpsspg.com/maps.htm

    lon = float(location[0:location.find(",")])
    lat = float(location[location.find(",") + 1:len(location)])
    a = 6378245.0  # 克拉索夫斯基椭球参数长半轴a
    ee = 0.00669342162296594323  # 克拉索夫斯基椭球参数第一偏心率平方
    PI = 3.14159265358979324  # 圆周率
    # 以下为转换公式
    x = lon - 105.0
    y = lat - 35.0
    # 经度
    dLon = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
    dLon += (20.0 * math.sin(6.0 * x * PI) + 20.0 * math.sin(2.0 * x * PI)) * 2.0 / 3.0
    dLon += (20.0 * math.sin(x * PI) + 40.0 * math.sin(x / 3.0 * PI)) * 2.0 / 3.0
    dLon += (150.0 * math.sin(x / 12.0 * PI) + 300.0 * math.sin(x / 30.0 * PI)) * 2.0 / 3.0
    # 纬度
    dLat = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    dLat += (20.0 * math.sin(6.0 * x * PI) + 20.0 * math.sin(2.0 * x * PI)) * 2.0 / 3.0
    dLat += (20.0 * math.sin(y * PI) + 40.0 * math.sin(y / 3.0 * PI)) * 2.0 / 3.0
    dLat += (160.0 * math.sin(y / 12.0 * PI) + 320 * math.sin(y * PI / 30.0)) * 2.0 / 3.0
    radLat = lat / 180.0 * PI
    magic = math.sin(radLat)
    magic = 1 - ee * magic * magic
    sqrtMagic = math.sqrt(magic)
    dLat = (dLat * 180.0) / ((a * (1 - ee)) / (magic * sqrtMagic) * PI)
    dLon = (dLon * 180.0) / (a / sqrtMagic * math.cos(radLat) * PI)
    wgsLon = lon - dLon
    wgsLat = lat - dLat
    return str(round(wgsLon, 6)) + ',' + str(round(wgsLat, 6))


if __name__ == '__main__':
    url = 'https://restapi.amap.com/v3/place/text?'
    params = {'keywords': '收费站', 'city': 'heilongjiang', 'page': '1', 'offset': 20,
              'key': 'eff48ee434d763609e59839fa946b9e1'}
    df = getstationinfo(url, params)

    df['wgsLocation'] = df['location'].map(GCJ2WGS)
    time = time.strftime('%Y.%m.%d',time.localtime(time.time()))
    save_path = 'F:\\18120900\\桌面\\%s-%s.xlsx'%(params['city'], time)
    df.to_excel(save_path, index=None)
    print("Done!")
