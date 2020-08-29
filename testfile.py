# -*- coding: utf-8 -*-
# CreatedTime: 2020/7/24 8:39
# Email: 1765471602@qq.com
# File: testfile.py
# Software: PyCharm
# Describe:

import time
import pandas as pd
import requests
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    # dealdata()
    sheet = pd.read_excel('河南高考一分一段表-2020.xlsx')
    fig, axes = plt.subplots(2, 1)
    axes[0].plot(sheet['score'], sheet['count'])
    axes[1].plot(sheet['score'], sheet['diff'])
    plt.tight_layout()
    plt.show()
    # for k in range(145):
    #     screenshot()
    #     print(k + 1, time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time())))
    #     time.sleep(600)
    print('Done!')
