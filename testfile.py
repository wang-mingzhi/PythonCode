# -*- coding: utf-8 -*-
# CreatedTime: 2020/7/24 8:39
# Email: 1765471602@qq.com
# File: testfile.py
# Software: PyCharm
# Describe:

import time
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib


def dealdata():
    data = pd.read_excel(r"F:\18120900\桌面\河南高考一分一段表-2020.xlsx")
    score = dict(zip(data.iloc[::2, 0], data.iloc[1::2, 0]))
    df = pd.DataFrame.from_dict(score, orient='index', columns=['count'])
    df = df.reset_index().rename(columns={'index': 'score'})
    df.sort_values(['score'], ascending=False, inplace=True)
    df['diff'] = df['count'].diff(1)
    df.iloc[0, 2] = df.iloc[0, 1]
    df.to_excel('河南高考一分一段表-2020.xlsx', index=None)


def screenshot():
    url = "http://restapi.amap.com/v3/staticmap?"
    params = {'location': '116.282104,39.915929', 'zoom': 14, 'size': '2048*2048',
              'scale': 2, 'traffic': 1, 'key': 'eff48ee434d763609e59839fa946b9e1'}
    r_text = requests.get(url, params=params)
    r_text.raise_for_status()  # 当出现错误时及时抛出错误
    filename = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time())) + ".png"
    with open("D:/拥堵区域变化趋势/" + filename, 'wb') as f:
        f.write(r_text.content)


def plotradarmap():
    labs = ['遵章\n驾驶率', '证照\n齐全率', '应急设施\n完善率', '价格', '违约补偿', '响应时间',
            '驾驶员\n执行效率', '守信率', '准时率', '准点率', '服务态度', '车辆内\n外环境']
    Y = [7.34, 7.09, 7.01, 6.75, 6.12, 6.86, 7.31, 7.58, 7.44, 7.46, 7.69, 7.53]

    # 获取 r 与 theta
    theta = np.linspace(0, 360, len(labs), endpoint=False)
    # 调整角度使得正中在垂直线上
    # theta += theta[-1] + 90 - 360
    # 将角度转化为单位弧度
    X_ticks = np.radians(theta)  # x轴标签所在的位置
    # 首尾相连
    X = np.append(X_ticks, X_ticks[0])
    Y = np.append(Y, Y[0])

    fig, ax = plt.subplots(figsize=(3.8, 3.8), subplot_kw=dict(projection='polar'))
    ax.plot(X, Y, 'ko-')
    ax.set_xticks(X[:-1])
    ax.set_xticklabels(labs, fontsize=10)
    # 设置背景坐标系
    ax.spines['polar'].set_visible(False)  # 将轴隐藏
    ax.grid(axis='y')  # 只有y轴设置grid
    # 设置X轴的grid
    n_grids = np.linspace(0, 10, 6)  # grid的网格数
    grids = [[i] * (len(X)) for i in n_grids]  # grids的半径
    for i, grid in enumerate(grids):  # 给grid 填充间隔色
        ax.plot(X, grid, color='grey', linewidth=0.5)
        if (i > 0) & (i % 2 == 0):
            ax.fill_between(X, grids[i], grids[i - 1], color='grey', alpha=0.1)
    plt.tight_layout()
    plt.show()


def main():
    # dealdata()
    # sheet = pd.read_excel('河南高考一分一段表-2020.xlsx')
    # fig, axes = plt.subplots(2, 1)
    # axes[0].plot(sheet['score'], sheet['count'])
    # axes[1].plot(sheet['score'], sheet['diff'])
    # plt.tight_layout()
    # plt.show()
    # for k in range(145):
    #     screenshot()
    #     print(k + 1, time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time())))
    #     time.sleep(600)
    plotradarmap()


if __name__ == "__main__":
    matplotlib.rcParams['font.family'] = 'simsun'
    matplotlib.rcParams['font.sans-serif'] = ['simsun']
    main()
    print('Done!')

