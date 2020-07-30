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
import matplotlib.pyplot as plt


def draw(data):
    data = geopandas.read_file(data)
    fig, ax = plt.subplots()
    data.to_crs({'init': 'epsg:4524'}).plot(ax=ax, alpha=0.85)  # 投影到epsg:4524
    plt.title("中国地图", fontsize=12)
    plt.tight_layout()
    plt.show()
    

def geojson2shape(file_path, file_save, crs):
    """
    geojson文件转存为shape文件
    @param file_path: geojson文件地址
    @param file_save: shape文件保存地址
    @param crs: 指定shape文件的坐标系统
    @return: None
    """
    try:
        data = geopandas.read_file(file_path)
        data.crs = crs
        data.to_file(file_save, driver='ESRI Shapefile', encoding='utf-8')
        print("保存成功，文件存放在："+file_save)
    except Exception as ex:
        print(ex)


if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['SimSun']
    plt.rcParams['axes.unicode_minus'] = False

    url = r"https://geo.datav.aliyun.com/areas_v2/bound/"
    r_text = requests.get(url + "100000.json")
    r_text.raise_for_status()  # 当出现错误时及时抛出错误
    content = json.loads(r_text.content)

    # 保存为json文件
    # with open(content['features'][0]['properties']['name'] + ‘.json’, 'w') as file:
    #     json.dump(content, file)
    draw(content)
    geojson2shape(content, '全国矢量地理文件.shp', {'init': 'epsg:4326'})
    print("Done")
