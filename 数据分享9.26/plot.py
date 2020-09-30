# -*- coding: utf-8 -*-
# @Time    : 2020/9/30 9:54
# @Author  : 1765471602@qq.com
# @File    : plot.py
# @Software: PyCharm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace
import mapclassify as mpc
import geopandas as gpd


def plot_flow(od_matrix, title, chengdu):
    colors = [cm.Spectral_r(x) for x in linspace(0, 1, 5)]
    df = od_matrix.copy()
    df_class = mpc.NaturalBreaks(df['ridership'])
    df['class'] = df_class.yb
    bins = df_class.bins
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    chengdu.plot(figsize=(5, 5), facecolor='none', edgecolor='black', linewidth=0.6, ax=ax, alpha=1)
    for i in range(len(bins)):
        sub = df[df['class'] == i]
        for j in sub.index:
            ax.annotate('', xy=(sub.loc[j, 'ori_x'], sub.loc[j, 'ori_y']),
                        xytext=(sub.loc[j, 'end_x'], sub.loc[j, 'end_y']),
                        arrowprops=dict(arrowstyle="->", color=colors[i], linewidth=0.5, alpha=0.8, shrinkA=0,
                                        shrinkB=0, patchA=None, patchB=None, connectionstyle='arc3,rad=-0.3'))
    ax.set_title(title, fontsize=16)
    ax.set_xlim(103.89, 104.24)
    ax.set_ylim(30.54, 30.8)
    line1, _ = ax.plot(((0, 0), (1, 1)), color=colors[0], alpha=0.8)
    line2, _ = ax.plot(((0, 0), (1, 1)), color=colors[1], alpha=0.8)
    line3, _ = ax.plot(((0, 0), (1, 1)), color=colors[2], alpha=0.8)
    line4, _ = ax.plot(((0, 0), (1, 1)), color=colors[3], alpha=0.8)
    line5, _ = ax.plot(((0, 0), (1, 1)), color=colors[4], alpha=1)
    plt.legend((line1, line2, line3, line4, line5),
               ['{} - {}'.format(0.0, round(bins[0], 1)),
                '{} - {}'.format(round(bins[0], 1), round(bins[1], 1)),
                '{} - {}'.format(round(bins[1], 1), round(bins[2], 1)),
                '{} - {}'.format(round(bins[2], 1), round(bins[3], 1)),
                '{} - {}'.format(round(bins[3], 1), round(bins[4], 1))],
               framealpha=1, title='Trips', facecolor='whitesmoke')
    plt.xticks([])
    plt.yticks([])
    # plt.savefig(name, dpi=300, transparent=True, pad_inches=0)
    # plt.close(fig=fig)
    plt.show()


def main():
    chengdu = gpd.read_file('成都街道/chengdu.shp')
    plot_flow(pd.read_csv('0小时od.csv'), '0:00', chengdu)
    # for i in range(24):
    #     hour = pd.read_csv('{}小时od.csv'.format(i))
    #     plot_flow(hour, '{}:00'.format(i), '{}:00'.format(i), chengdu)


if __name__ == '__main__':
    main()
    print('Done!')
