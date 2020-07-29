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
import matplotlib.ticker as ticker


def downloadjson(params):
    r_text = requests.get(url + params)
    r_text.raise_for_status()  # 当出现错误时及时抛出错误
    content = json.loads(r_text.content)
    file_name = content['features'][0]['properties']['name']
    path = str(file_name) + ".json"
    with open(path, 'w') as file:
        json.dump(content, file)


def draw():
    data = geopandas.read_file('全国.json')
    fig, ax = plt.subplots()
    data.to_crs({'init': 'epsg:4524'}).plot(ax=ax, alpha=1)  # 投影到epsg:4524
    plt.title("中国地图", fontsize=12)
    plt.tight_layout()
    plt.show()
    

def saveshapefile(file_path, file_save):
    try:
        data = geopandas.read_file(file_path)
        data.crs = {'init': 'epsg:4326'}
        data.to_file(file_save, driver='ESRI Shapefile', encoding='utf-8')
        print("--保存成功，文件存放位置："+file_save)
    except Exception as ex:
        print(ex)


if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['SimSun']
    plt.rcParams['axes.unicode_minus'] = False
    url = r"https://geo.datav.aliyun.com/areas_v2/bound/"
    # downloadjson("100000.json")
    # saveshapefile('全国.json', '全国矢量地理文件.shp')
    draw()
    print("Done")
