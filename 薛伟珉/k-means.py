from numpy import *
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt
 # 加载数据
def loadDataSet(filename):  # 解析文件，按tab分割字段，得到一个浮点数字类型的矩阵
     dataMat = []              # 文件的最后一个字段是类别标签
     fr = open(filename)
     for line in fr.readlines():
         curLine = line.strip().split('\t')
         fltLine = list(map(float, curLine))    # 将每个元素转成float类型
         dataMat.append(fltLine)

     return dataMat

# print(loadDataSet("three1.txt"))

# #  # 计算欧几里得距离
# def distEclud(vecA, vecB):
#      return sqrt(sum(power(vecA - vecB, 2))) # 求两个向量之间的距离

def distEclud(vecA, vecB):
    lng1 = vecA[0,0]
    lat1 = vecA[0,1]
    lng2 = vecB[0,0]
    lat2 = vecB[0,1]
    lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    dis = 2 * asin(sqrt(a)) * 6371 * 1000
    dis = round(dis / 1000, 3)
    return dis  # 求两个向量之间的距离

 # 构建聚簇中心，取k个(此例中为4)随机'质心
 #是一个326*2的矩阵
def randCent(dataSet, k):
     n = shape(dataSet)[1]#查看矩阵的维数
     dataSet = mat(dataSet)
     centroids = mat(zeros((k,n)))   # 每个质心有n个坐标值，总共要k个质心#将目标数据类型转换为矩阵
     for j in range(n):
         minJ = min(dataSet[:,j])
         maxJ = max(dataSet[:,j])
         rangeJ = float(maxJ - minJ)
         centroids[:,j] = minJ + rangeJ * random.rand(k, 1)
     return centroids

# print(randCent(loadDataSet("three1.txt"),4))

 # k-means 聚类算法
def kMeans(dataSet, k, distMeans =distEclud, createCent = randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))    # 用于存放该样本属于哪类及质心距离
    # clusterAssment第一列存放该数据所属的中心点，第二列是该数据到中心点的距离
    centroids = createCent(dataSet, k)
    clusterChanged = True   # 用来判断聚类是否已经收敛
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # 把每一个数据点划分到离它最近的中心点
            minDist = inf
            minIndex = -1#y一开始最小距离是无穷大，它属于-1这个聚类中心
            for j in range(k):
                distJI = distMeans(centroids[j,:], dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j  # 如果第i个数据点到第j个中心点更近，则将i归属为j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True;  # 如果分配发生变化，则需要继续迭代
                clusterAssment[i,:] = minIndex,minDist**2   # 并将第i个数据点的分配情况存入字典
        # print(centroids)
        for cent in range(k):   # 重新计算中心点
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]   # 去第一列等于cent的所有列
            centroids[cent,:] = mean(ptsInClust, axis = 0)  # 算出这些数据的中心点
    return centroids, clusterAssment
# 计算聚类内距离




datMat = mat(loadDataSet('聚类中心数据.txt'))
# myCentroids_1,clustAssing_1 = kMeans(datMat,1)
myCentroids,clustAssing = kMeans(datMat,15)
#
def jlnjl(k,juzhen):#求内部距离
    dn=0
    q=0
    for j in range(len(juzhen)):
        if juzhen[j,0]==k:
            dn=dn +juzhen[j,1]
            q = q + 1
    # print(dn,q)
    return dn,q

def jlwjl(juzhen1):#求外部距离
    a = 0
    for i in range(len(myCentroids)):
        a = a + distEclud(myCentroids[0], myCentroids[i])
    return a

print('外部距离和=',jlwjl(myCentroids))

print(myCentroids)
# print(clustAssing)
# print(myCentroids[:,0])
# print(myCentroids[:,1])
# print(clustAssing[:,0])
# print(clustAssing[:,1])
# print(data)
k = 15
a = 0
for i in range(k):
    print(i)
    print(jlnjl(i,clustAssing))
    a = a + jlnjl(i,clustAssing)[0]
print('所有内部距离和=',a)
# print(jlnjl(0,clustAssing),jlnjl(1,clustAssing),jlnjl(2,clustAssing),jlnjl(3,clustAssing),jlnjl(4,clustAssing))
# print(jlnjl(0,clustAssing)+jlnjl(1,clustAssing)+jlnjl(2,clustAssing)+jlnjl(3,clustAssing)+jlnjl(4,clustAssing))
# print(clustAssing[2,1])


# color='rgbycmykw'
# color= ['b','g','c','m','y','k','orange','olive','sienna','greenyellow','deepskyblue','slateblue','teal','pink','grey','purple','aqua',"tomato",'chocolate']
color= ['aqua',
'aquamarine',
'bisque',
'black',
'blue',
'blueviolet',
'brown',
'burlywood',
'cadetblue',
'chartreuse',
'chocolate',
'coral',
'cornflowerblue',
'crimson',
'cyan',
'darkblue',
'darkcyan',
'darkgoldenrod',
'darkgray',
'darkgreen',
'darkkhaki',
'darkmagenta',
'darkolivegreen',
'darkorange',
'darkorchid',
'darkred',
'darksalmon',
'darkseagreen',
'darkslateblue',
'darkslategray',
'darkturquoise',
'darkviolet',
'deeppink',
'deepskyblue',
'dimgray',
'dodgerblue',
'firebrick',
'forestgreen',
'gainsboro',
'gold',
'goldenrod',
'gray',
'green',
'greenyellow',
'hotpink',
'indianred',
'indigo',
'khaki',
'lawngreen',
'lime',
'limegreen',
'magenta',
'maroon',
'mediumaquamarine',
'mediumblue',
'mediumorchid',
'mediumpurple',
'mediumseagreen',
'mediumslateblue',
'mediumspringgreen',
'mediumturquoise',
'mediumvioletred',
'midnightblue',
'moccasin',
'navajowhite',
'navy',
'olive',
'olivedrab',
'orange',
'orangered',
'orchid',
'palegoldenrod',
'palegreen',
'paleturquoise',
'palevioletred',
'peachpuff',
'peru',
'pink',
'plum',
'powderblue',
'purple',
'red',
'rosybrown',
'royalblue',
'saddlebrown',
'salmon',
'sandybrown',
'seagreen',
'sienna',
'silver',
'skyblue',
'slateblue',
'slategray',
'springgreen',
'steelblue',
'tan',
'teal',
'thistle',
'tomato',
'turquoise',
'violet',
'wheat',
'yellow',
'yellowgreen',
'coral',
'cornflowerblue',
'crimson',
'cyan',
'darkblue',
'darkcyan',
'darkgoldenrod',
'darkgray',
'darkgreen',
'darkkhaki',
'sienna',
'silver',
'skyblue',
'slateblue',
'slategray',
'springgreen',
'steelblue',
'tan',
'teal',
'thistle',
'tomato',
'turquoise',
'violet',
'wheat',
'mediumaquamarine',
'mediumblue',
'mediumorchid',
'mediumpurple',
'mediumseagreen',
'mediumslateblue',
'black',
'blue',
'blueviolet',
'brown',
'burlywood',
'cadetblue',
'chartreuse',
'chocolate',
'coral',
'cornflowerblue',
'crimson',
'springgreen',
'steelblue',
'cornflowerblue',
'crimson',
'cyan',
'darkblue'
]

dataMat = mat(loadDataSet('聚类中心数据.txt'))
plt.subplot(2,2,1)#第一张：dataMat
fig1 = plt.figure(num='顾客群聚集点', figsize=(5,5), dpi=75)
for i in range(len(dataMat)):
    # plt.xlim(116.38,116.46)
    plt.plot(dataMat[i,0],dataMat[i,1],"ro",linewidth=0.5, markersize=3)
plt.show()

# plt.subplot(2, 2, 2)  # 第二张:myCentroids
# for i in range(len(myCentroids)):
#     plt.plot(myCentroids[i, 0], myCentroids[i, 1], "y*")


plt.subplot(2,2,4)#第四张：dataMat聚类结果染色图
fig1 = plt.figure(num='顾客群聚集点', figsize=(5,5),dpi=100)
for i in range(len(dataMat)):
    plt.plot(dataMat[i,0],dataMat[i,1],"ro",color=color[int(clustAssing[i,0])],linewidth=0.5, markersize=3)
for i in range(len(myCentroids)):
    plt.plot(myCentroids[i, 0], myCentroids[i, 1], "r*")
plt.show()

# plt.subplot(2, 2, 2)  # 第二张:myCentroids
# for i in range(len(myCentroids)):
#     plt.plot(myCentroids[i, 0], myCentroids[i, 1], "y*")


plt.subplot(2, 2, 3)  # 第三张：聚类结果图
for i in range(len(clustAssing)):
    plt.plot(clustAssing[i, 0], clustAssing[i, 1], "ro", color=color[int(clustAssing[i, 0])])
plt.show()


# plt.scatter(dataMat[],y)
# plt.scatter(centers[:,0],centers[:,1],marker="*",s=500,c="r")
# plt.show()

# plt.scatter(x,y)
# plt.scatter(centers[:,0],centers[:,1],marker="*",s=500,c="r")
# plt.show()
