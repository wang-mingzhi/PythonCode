import numpy as np  # 数据结构
import sklearn.cluster as skc  # 密度聚类
from sklearn import metrics   # 评估模型
import matplotlib.pyplot as plt  # 可视化绘图
from math import radians, cos, sin, asin, sqrt
import pandas as pd
import xlrd

# def loadDataSet(filename):  # 解析文件，按tab分割字段，得到一个浮点数字类型的矩阵
#      dataMat = []              # 文件的最后一个字段是类别标签
#      fr = open(filename)
#      for line in fr.readlines():
#          curLine = line.strip().split('\t')
#          fltLine = list(map(float, curLine))    # 将每个元素转成float类型
#          dataMat.append(fltLine)
#
#      return dataMat
data = []
da = xlrd.open_workbook('three4.xlsx')
table = da.sheets()[0]
cols_0 = table.col_values(0)
cols_1 = table.col_values(1)
nrows = table.nrows
# print(nrows)
# print(table.cell_value(1,1) )

for i in range(nrows):
    m = [0,0]
    m[0]=cols_0[i]
    m[1]=cols_1[i]
    data.append(m)
# print(data)
# data=[
#     [-2.68420713,1.469732895],[-2.71539062,-0.763005825],[-2.88981954,-0.618055245],[-2.7464372,-1.40005944],[-2.72859298,1.50266052],
#     [-2.27989736,3.365022195],[-2.82089068,-0.369470295],[-2.62648199,0.766824075],[-2.88795857,-2.568591135],[-2.67384469,-0.48011265],
#     [-2.50652679,2.933707545],[-2.61314272,0.096842835],[-2.78743398,-1.024830855],[-3.22520045,-2.264759595],[-2.64354322,5.33787705],
#     [-2.38386932,6.05139453],[-2.6225262,3.681403515],[-2.64832273,1.436115015],[-2.19907796,3.956598405],[-2.58734619,2.34213138],
#     [1.28479459,3.084476355],[0.93241075,1.436391405],[1.46406132,2.268854235],[0.18096721,-3.71521773],[1.08713449,0.339256755],
#     [0.64043675,-1.87795566],[1.09522371,1.277510445],[-0.75146714,-4.504983795],[1.04329778,1.030306095],[-0.01019007,-3.242586915],
#     [-0.5110862,-5.681213775],[0.51109806,-0.460278495],[0.26233576,-2.46551985],[0.98404455,-0.55962189],[-0.174864,-1.133170065],
#     [0.92757294,2.107062945],[0.65959279,-1.583893305],[0.23454059,-1.493648235],[0.94236171,-2.43820017],[0.0432464,-2.616702525],
#     [4.53172698,-0.05329008],[3.41407223,-2.58716277],[4.61648461,1.538708805],[3.97081495,-0.815065605],[4.34975798,-0.188471475],
#     [5.39687992,2.462256225],[2.51938325,-5.361082605],[4.9320051,1.585696545],[4.31967279,-1.104966765],[4.91813423,3.511712835],
#     [3.66193495,1.0891728],[3.80234045,-0.972695745],[4.16537886,0.96876126],[3.34459422,-3.493869435],[3.5852673,-2.426881725],
#     [3.90474358,0.534685455],[3.94924878,0.18328617],[5.48876538,5.27195043],[5.79468686,1.139695065],[3.29832982,-3.42456273]
# ]

# data = pd.read_table("three3.txt", sep = "\t")

X = np.array(data)

def get_distance(array_1, array_2):
    lon_a = array_1[0]
    lat_a = array_1[1]
    lon_b = array_2[0]
    lat_b = array_2[1]
    radlat1 = radians(lat_a)
    radlat2 = radians(lat_b)
    a = radlat1 - radlat2
    b =radians(lon_a) - radians(lon_b)
    s = 2 * asin(sqrt(pow(sin(a/2),2) +cos(radlat1) * cos(radlat2)*pow(sin(b/2),2)))
    earth_radius = 6378137
    s = s * earth_radius
    return s



db = skc.DBSCAN(eps=70, min_samples=2, metric=get_distance).fit(X) #DBSCAN聚类方法 还有参数，matric = ""距离计算方法
labels = db.labels_  #和X同一个维度，labels对应索引序号的值 为她所在簇的序号。若簇编号为-1，表示为噪声

print('每个样本的簇标号:')
print(labels)

raito = len(labels[labels[:] == -1]) / len(labels)  #计算噪声点个数占总数的比例
print('噪声比:', format(raito, '.2%'))
unique_labels = set(labels)


colors = ['b','g','c','m','y','k','orange','olive','sienna','greenyellow','deepskyblue','slateblue','teal','pink','grey','purple','aqua',"tomato",'chocolate']


n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目

print('分簇的数目: %d' % n_clusters_)
print("轮廓系数: %0.3f" % metrics.silhouette_score(X, labels)) #轮廓系数评价聚类的好坏

def zhongxin(l):
    min = 999999
    for i in range(len(l)):
        a = 0
        for j in range(len(l)):
            a = a + get_distance(l[i], l[j])
        if a<= min:
            min = a
            m = i
    return one_cluster[m]

d = 0
g = 0
for i in range(n_clusters_):
    print('簇 ', i, '的所有样本:')
    one_cluster = X[labels == i]
    print(one_cluster)
    print('簇 ', i, '的长度:',len(one_cluster))
    print(zhongxin(one_cluster))
    g = g + get_distance(zhongxin(one_cluster),[116.4388828, 39.93785768])
    print('第',i,'次聚类中心到整体中心距离求和', g)
    c = 0
    for j in range(len(one_cluster)):
        c = c + get_distance(zhongxin(one_cluster), one_cluster[j])
    print('簇', i, '的内部距离', c)
    d = d + c
    print('各类聚类间内部距离求和：',d)
    print('聚类中心到整体中心距离求和',g)
    plt.plot(one_cluster[:,0],one_cluster[:,1],'o')

# for k, col in zip(unique_labels, colors):
#     if k == -1:  # 聚类结果为-1的样本为离散点
#         # 使用黑色绘制离散点
#         col = ['grey']

print('簇 ', -1, '的所有样本:')
one_cluster = X[labels == -1]
print(one_cluster)
print('簇 ', -1, '的长度:',len(one_cluster))
print(zhongxin(one_cluster))
plt.plot(one_cluster[:,0],one_cluster[:,1],'*k')

plt.show()


#[116.4388828   39.93785768]

#print(data)
