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


def crawler(url, code, crs=None):
    """
    爬取url数据并返回GeoDataFrame格式数据
    @param url: url
    @param code: 待爬取城市的代码，与url拼合后爬取数据
    @param crs: GeoDataFrame中数据所在的地理坐标系，默认为：{'init': 'epsg:4326'}
    @return: GeoDataFrame
    """
    if crs is None:
        crs = {'init': 'epsg:4326'}
    r_text = requests.get(url + code + '.json')
    r_text.raise_for_status()  # 当出现错误时及时抛出错误
    content = json.loads(r_text.content)  # 解析url返回的数据
    columns, properties, geometry = "", [], []
    for item in content['features']:
        columns = item['properties'].keys() if columns == "" else columns  # 获取标题
        # 获取对应值,并把列表中的值全部转为str否则生成shape文件时会存在问题
        properties.append([str(item) for item in list(item['properties'].values())])
        polygons = [Polygon(coordinate[0]) for coordinate in item['geometry']['coordinates']]
        geometry.append(MultiPolygon(polygons))
    geofencedf = pd.DataFrame(properties, columns=columns)
    result = geopandas.GeoDataFrame(geofencedf, geometry=geometry)
    result.crs = crs
    return result


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
    data.to_file(file_save, driver='ESRI Shapefile', encoding='utf-8')
    print("保存成功，文件存放在：" + file_save)


if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['SimSun']
    plt.rcParams['axes.unicode_minus'] = False

    areacode = '100000_full'
    gdf = crawler(r"https://geo.datav.aliyun.com/areas_v2/bound/", areacode)
    # 保存为json文件
    # with open(areacode + ".json", 'w') as file:
    #     json.dump(content, file)
    draw(gdf)
    geojson2shape(gdf, areacode + '.shp', {'init': 'epsg:4326'})
    print("Done")
