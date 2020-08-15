# -*- coding: utf-8 -*-
# CreatedTime: 2020/7/29 8:34
# Email: 1765471602@qq.com
# File: 获取省市县渠化.py
# Software: PyCharm
# 参考连接：https://mp.weixin.qq.com/s/cUW7cm0_shipSs2_-3x5Ag
# 参考连接：https://mp.weixin.qq.com/s/JKP-Do8zR_hiW4qJrahgYQ

import requests
import json
import geopandas
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import MultiPolygon, Polygon


def crawler(_columns, _properties, _geometry, url, _code):
    """
    爬取url数据并返回_columns, _properties, _geometry
    @param _columns: 用来存放列名
    @param _geometry: 用来存放每个区划的polygon
    @param _properties: 每个区划的相关信息
    @param url: url
    @param _code: 待爬取城市的代码，与url拼合后爬取数据
    @return: _columns, _properties, _geometry
    """
    r_text = requests.get(url + _code + '.json')
    r_text.raise_for_status()  # 当出现错误时及时抛出错误
    content = json.loads(r_text.content)  # 解析url返回的数据
    for item in content['features']:
        if not _columns:
            _columns.append(list(item['properties'].keys()))
        # 获取对应值,并把列表中的值全部转为str否则生成shape文件时会存在问题
        _properties.append([str(item) for item in list(item['properties'].values())])
        polygons = [Polygon(coordinate[0]) for coordinate in item['geometry']['coordinates']]
        _geometry.append(MultiPolygon(polygons))


def draw(data):
    fig, ax = plt.subplots()
    data.to_crs({'init': 'epsg:4524'}).plot(ax=ax, alpha=0.85)  # 投影到epsg:4524,避免看起来扁
    plt.title("中国地图", fontsize=12)
    plt.tight_layout()
    plt.show()
    

def geojson2shape(data, file_save, crs):
    """
    geojson文件转存为shape文件
    @param data: GeoDataFrame格式数据
    @param file_save: shape文件保存地址
    @param crs: 指定shape文件的坐标系统
    @return: None
    """
    data.to_crs(crs, inplace=True)
    data.to_file(file_save + '.shp', driver='ESRI Shapefile', encoding='utf-8')
    print("保存成功，文件存放在：" + file_save)


def geojson2file(data, file_save, crs):
    data.to_crs(crs, inplace=True)
    data.to_json(file_save + '.json')
    print("保存成功，文件存放在：" + file_save)


if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['SimSun']
    plt.rcParams['axes.unicode_minus'] = False

    columns, properties, geometry = [], [], []
    areacode_list = ['410000_full']
    for code in areacode_list:
        crawler(columns, properties, geometry, r"https://geo.datav.aliyun.com/areas_v2/bound/", code)
    df = pd.DataFrame(properties, columns=columns[0])
    gdf = geopandas.GeoDataFrame(df, geometry=geometry)
    gdf.crs = {'init': 'epsg:4326'}  # 设置geojson的地理坐标系
    draw(gdf)                        # 画出geojson地图
    # file_name = '省级行政区划'
    # geojson2shape(gdf, file_name, {'init': 'epsg:4326'})  # 转存为shape文件
    # print("Done")
