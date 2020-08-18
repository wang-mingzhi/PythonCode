# -*- coding: utf-8 -*-
# CreatedTime: 2020/7/24 8:39
# Email: 1765471602@qq.com
# File: testfile.py
# Software: PyCharm
# Describe:
import time
import pandas as pd
import matplotlib.pyplot as plt


def fun():
    a = 0
    b = 0
    for i in range(100000):
        a = a + i * i
    for i in range(3):
        b += 1
        time.sleep(0.1)
    return a + b

 # 注释
def dealdata():
    data = pd.read_excel(r"F:\18120900\桌面\河南高考一分一段表-2020.xlsx")
    score = dict(zip(data.iloc[::2, 0], data.iloc[1::2, 0]))
    df = pd.DataFrame.from_dict(score, orient='index', columns=['count'])
    df = df.reset_index().rename(columns={'index': 'score'})
    df.sort_values(['score'], ascending=False, inplace=True)
    df['diff'] = df['count'].diff(1)
    df.iloc[0, 2] = df.iloc[0, 1]
    df.to_excel('河南高考一分一段表-2020.xlsx', index=None)


if __name__ == "__main__":
    # fun()
    dealdata()
    sheet = pd.read_excel('河南高考一分一段表-2020.xlsx')
    fig, axes = plt.subplots(2, 1)
    axes[0].plot(sheet['score'], sheet['count'])
    axes[1].plot(sheet['score'], sheet['diff'])
    plt.tight_layout()
    plt.show()
    print('Done!')
