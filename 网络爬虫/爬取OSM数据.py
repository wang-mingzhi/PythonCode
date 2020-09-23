# -*- coding: utf-8 -*-
# CreatedTime: 2020/9/12 21:09
# Email: 1765471602@qq.com
# File: 爬取OSM数据.py
# Software: PyCharm
# Describe: 

import osmnx as ox


def crawler():
    city = ox.graph_from_place("福田区,深圳")  # 从OSM上爬取福田区地图
    ox.plot_graph(city)  # 用python展示地图
    ox.save_graph_shapefile(city, filepath='szft')  # 保存地图


def main():
    crawler()


if __name__ == "__main__":
    main()
    print('Done')
