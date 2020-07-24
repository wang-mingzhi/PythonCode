# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 13:29:46 2020
高速公路客货运距分析研究-以宁夏回族自治区为例
@author: 18120900
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
from scipy.interpolate import make_interp_spline
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics.pairwise import cosine_similarity
from minepy import MINE
from scipy import stats
from PIL import Image
import io


class SeabornFig2Grid:

    def __init__(self, seaborngrid, fig, subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n, m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i, j], self.subgrid[i, j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h = self.sg.ax_joint.get_position().height
        h2 = self.sg.ax_marg_x.get_position().height
        r = int(np.round(h / h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r + 1, r + 1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        # https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure = self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())


def dealdata(data, groupby, index):
    pivoted_data = (data
                    .groupby(groupby).sum().reset_index()
                    .assign(avgmiles=lambda x: round(x['totalmiles'] / x['count'], 2))
                    .pivot_table(values='avgmiles', index=index, columns=['veh'], aggfunc=np.mean)
                    .reset_index())
    return pivoted_data


def statistic(data):
    data.describe()
    data.skew()
    data.kurt()


def drawyeartrendgram(data, lblname):
    f, axes = plt.subplots(3, 2.5, sharex='all', sharey='all', figsize=(5.12, 4))
    plt.subplots_adjust(0.07, 0.07, 0.98, 0.98, 0, 0)
    for i in range(len(lblname)):
        y = data.iloc[i, 2:]
        y = (y - y.min()) / (y.max() - y.min())
        axes[i // 3, i % 3].plot(data.columns[2:], y, 'k-', label=lblname[i])
        axes[i // 3, i % 3].legend(prop=font_CN, loc='upper right', frameon=False)
    seticksandlegend(loc='upper right', frameon=False)
    savefig2tif(f, '各车型变化趋势图.tif')


def hourtrendgram(pivot_data, lblname):
    f = plt.figure(figsize=(2.6, 2.5))
    ax = plt.axes()
    plt.subplots_adjust(0.1, 0.17, 0.98, 0.98, 0.2, 0.2)
    x = pivot_data.iloc[:, 0]
    x_new = np.linspace(x.min(), x.max(), 300)
    for i in range(len(lblname)):
        y = pivot_data.iloc[:, i + 1]
        y_smooth = make_interp_spline(x, y)(x_new)
        plt.plot(x_new, y_smooth, color='k', linestyle=dict_linestyle[i], linewidth=1, label=lblname[i])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    seticksandlegend('Hour/h', 'Miles/km', loc='upper right', ncol=2, frameon=False)
    savefig2tif(f, '小时趋势变化图.tif')


def hourlineplot(data, lblname):
    f = plt.figure(figsize=(2.6, 2.5))
    ax = plt.axes()
    plt.subplots_adjust(0.18, 0.15, 0.98, 0.85, 0.2, 0.2)
    r1 = [1, 2, 3, 4] if len(lblname) == 4 else [2, 3, 4, 5, 6]
    data['veh'] = data['veh'].replace(r1, lblname)
    data['avgmiles'] = round(data['totalmiles'] / data['count'], 2)
    veh = '客车' if len(lblname) == 4 else '货车'

    sns.lineplot('time', 'avgmiles', style='veh', color='k', data=data)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('Hour/h', fontdict=font_EN)
    plt.ylabel('Miles/km', fontdict=font_EN)
    plt.legend(prop=font_CN,
               loc='lower left',
               ncol=3,
               frameon=False,
               columnspacing=0.5,
               bbox_to_anchor=(0, 1),
               borderaxespad=0)
    savefig2tif(f, veh + '平均运距小时变化趋势图.tif')


def drawfig(pivot_data, lblname):
    veh = '客车' if len(lblname) == 4 else '货车'

    f = plt.figure(figsize=(3, 2.2))
    ax = plt.axes()
    plt.subplots_adjust(0.15, 0.2, 0.98, 0.98, 0.2, 0.25)
    for i in range(len(lblname)):
        plt.plot(pivot_data.iloc[:, 0],
                 pivot_data.iloc[:, i + 1].rolling(7).mean(),
                 color='k',
                 linestyle=dict_linestyle[i],
                 linewidth=1,
                 label=lblname[i])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    loc = 'best' if veh == '货车' else 'upper center'
    seticksandlegend(xlabel='日期', ylabel='平均行驶里程/km', ncol=-(-len(lblname) // 2), loc=loc, rotation=15)
    savefig2tif(f, veh + '平均运距年度变化趋势图.tif')

    f = plt.figure(figsize=(3, 2))
    ax = plt.axes()
    plt.subplots_adjust(0.17, 0.17, 0.96, 0.96, 0.2, 0.25)
    for i in range(len(lblname)):
        sns.distplot(pivot_data.iloc[:, i + 1],
                     hist=False,
                     kde_kws={"shade": False, "linestyle": dict_linestyle[i]},
                     color='k',
                     label=lblname[i])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    seticksandlegend(xlabel='平均行驶里程/km', ylabel='频率', loc='upper right')
    savefig2tif(f, veh + '平均运距年度分布图.tif')


def drawKDEfig(data, lblname):
    sns.set_style(style='white')
    fig = plt.figure(figsize=(5.12, 4))
    gs = gridspec.GridSpec(2, 3, None, 0.1, 0.1, 0.98, 0.98, 0.2, 0.25)
    for i in range(len(lblname)):
        temp_data = data if i == 0 else data[data.veh == i + 1]
        axes = sns.jointplot(x='miles',
                             y='weight',
                             data=temp_data,
                             kind='kde',
                             color='k',
                             xlim=(-50, 500),
                             height=4,
                             space=0)
        ylabel = 'weight/t' if i in [0, 3] else ''
        axes.set_axis_labels(lblname[i] + '\'s haul distance/km', ylabel, fontdict=font_EN)
        SeabornFig2Grid(axes, fig, gs[i])
    seticksandlegend()
    savefig2tif(fig, '货重和运距联合密度分布图.tif')


def drawbarplot(data):
    f = plt.figure(figsize=(3, 2.5))
    ax = plt.axes()
    plt.subplots_adjust(0.18, 0.15, 0.98, 0.85, 0.2, 0.2)
    tick_label = data.iloc[:, 0]
    x = np.arange(len(tick_label))
    width = 0.3
    ax.bar(x - width/2, data.iloc[:, 1], width=width, color='white', edgecolor='k', hatch='///', label='宁夏')
    ax.bar(x + width/2, data.iloc[:, 2], width=width, color='white', edgecolor='k', hatch='+++', label='黑龙江')
    ax.set_ylabel('平均行驶里程/km', fontdict=font_CN)
    ax.set_xlabel('车型', fontdict=font_CN)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_label)
    ax.legend(prop=font_CN)


def seticksandlegend(xlabel='', ylabel='', loc='upper center', ncol=1, frameon=False, rotation=0):
    plt.xlabel(xlabel, fontdict=font_CN)
    plt.ylabel(ylabel, fontdict=font_CN)
    plt.xticks(fontproperties='Times New Roman', size=9, rotation=rotation)
    plt.yticks(fontproperties='Times New Roman', size=9)
    plt.legend(prop=font_CN,
               loc=loc,
               ncol=ncol,
               frameon=frameon,
               columnspacing=0.5,
               labelspacing=0.2,
               borderaxespad=0)


def savefig2tif(fig, filename):
    png1 = io.BytesIO()
    fig.savefig(png1, format="png")
    Image.open(png1).save(filename)
    png1.close()


def calMIC(data):
    for i in range(5):
        mine = MINE(alpha=0.6, c=15)
        miles = data[data.veh == (i + 2)].iloc[:, 1]
        weight = data[data.veh == (i + 2)].iloc[:, 2]
        mine.compute_score(miles, weight)
        print("Without noise:", "MIC", mine.mic())


def correlationanalysis(data):
    _, axes = plt.subplots(2, len(data.columns) - 1, sharex='all', sharey='all', figsize=(5.12, 4))
    plt.subplots_adjust(0.12, 0.07, 0.97, 0.97, 0, 0)
    for i in range(len(data.columns) - 1):
        plot_acf(data.iloc[:, i + 1], ax=axes[0, i], title='')
        plot_pacf(data.iloc[:, i + 1], ax=axes[1, i], title='')
    axes[0, 0].set_ylabel('ACF', fontdict=font_EN)
    axes[1, 0].set_ylabel('PACF', fontdict=font_EN)


def divisionoftimeperiod(data):
    """
    时段划分模型
    @param data: 数据集
    @return: None
    """
    title = data.columns.values.tolist()
    # 生成21*21的矩阵(7天，每天三个纬度)，即一天与其他天的余弦值
    result = cosine_similarity(data.values, data.values)
    df = pd.DataFrame(columns=title, data=result, index=title)
    plt.figure(figsize=(5, 4))
    plt.subplots_adjust(0.09, 0.1, 0.98, 0.98, 0.2, 0.2)
    sns.heatmap(df, cmap='coolwarm')
    plt.show()


def calSimilarity(p, q):
    assert len(p) == len(q), "两个向量的维数不同"
    print('余弦相似性', '欧氏距离', '皮尔逊相关系数')
    for i in range(len(p)):
        cc = cosine_similarity(p[i].reshape(1, -1), q[i].reshape(1, -1))[0]
        oo = np.sqrt(np.sum(np.square(p[i] - q[i])))
        pp = stats.pearsonr(p[i], q[i])
        print(cc, oo, pp)


if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['simsun']  # 中文字体设置-黑体
    plt.rcParams['xtick.direction'] = 'in'  # 设置x轴刻度线方向，朝内还是朝外
    plt.rcParams['ytick.direction'] = 'in'  # 设置y轴刻度线方向，朝内还是朝外
    plt.rcParams['lines.linewidth'] = 1  # 设置线宽
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    font_EN = {'family': 'Times New Roman', 'weight': 'normal', 'size': 9}
    font_CN = {'family': 'simsun', 'weight': 'normal', 'size': 9}
    dict_markers = {0: 'x', 1: 'D', 2: '^', 3: 'o', 4: 's'}
    dict_linestyle = {0: '-', 1: '--', 2: '-.', 3: ':', 4: (0, (5, 5))}
    label = ['1类', '2类', '3类', '4类', '二轴', '三轴', '四轴', '五轴', '六轴']
    # 读取数据
    datafilepath = "Data.xlsx"
    sheet = pd.read_excel(datafilepath, [0, 1, 2, 3, 4, 5, 6, 7])
    #
    # 计算统计量
    # statistic(dealdata(sheet[1], ['date', 'veh'], ['date']))  # 客车
    # statistic(dealdata(sheet[2], ['date', 'veh'], ['date']))  # 货车
    #
    # # 画各个车型从2014到2018的变化趋势图
    # drawyeartrendgram(sheet[0], label)
    #
    # 各个车型年度变化趋势图、平均运距直方图
    drawfig(dealdata(sheet[1], ['date', 'veh'], ['date']), label[:4])  # 客车
    drawfig(dealdata(sheet[2], ['date', 'veh'], ['date']), label[4:])  # 货车
    #
    # # 各个车型小时变化趋势图
    # hourtrendgram(dealdata(sheet[1], ['time', 'veh'], ['time']), label[:4])  # 客车-平滑曲线
    # hourtrendgram(dealdata(sheet[2], ['time', 'veh'], ['time']), label[4:])  # 货车-平滑曲线
    # hourlineplot(sheet[1], label[:4])                                        # 客车-含有置信区间
    # hourlineplot(sheet[2], label[4:])                                        # 货车*含有置信区间
    #
    # # 黑龙江和宁夏结果对比
    # drawbarplot(sheet[4])
    #
    # # 相关性分析图
    # correlationanalysis(dealdata(sheet[1], ['date', 'veh'], ['date']))  # 客车
    # correlationanalysis(dealdata(sheet[2], ['date', 'veh'], ['date']))  # 货车
    #
    # # 货车平均运距与载重的kde图
    # drawKDEfig(sheet[3], ['All Truck', 'Axis-II', 'Axis-III', 'Axis-IV', 'Axis-V', 'Axis-VI'])
    # # 计算车货总重和平均运距之间的互相关信息MIC
    # calMIC(sheet[3])
    #
    # # 计算余弦相似性
    # divisionoftimeperiod(sheet[5])
    #
    # 计算变量之间的相似性
    # calSimilarity(sheet[6].values[:, 1:], sheet[7].values[:, 1:])

    plt.show()
