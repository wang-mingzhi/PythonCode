#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#-- coding: utf-8 --
# import xlrd
# import xlwt
import numpy
from numpy import *
from math import radians, cos, sin, asin, sqrt
import sys
import matplotlib.pyplot as plt

#末端网点备选点
bxd = array([[116.4118015,39.97084263],[116.4013755,39.96901633],[116.4057306,39.96135127],[116.4173983,39.96026984],[116.3953146,39.95762957],[116.4298179,39.95657852],[116.3849251,39.95370354],[116.3939942,39.94941589],[116.4429649,39.94718116],[116.4156607,39.94509221],[116.4007794,39.93996155],[116.4382709,39.9369109],[116.3952922,39.93192699],[116.428873,39.9288255],[116.4108311,39.92814972]])

#末端网点距离配送中心距离
bxd_jl = [15.42903052,16.049895,15.28403744,14.37635372,15.85199028,13.26637604,16.4469638,15.54528056,11.75812576,13.67324192,14.59564904,11.54366416,14.72652988,11.91454568,13.33762248]

#用户需求点
xqd = array([[116.41799286,39.96252571],
            [116.44943,39.94881367],
            [116.44568211,39.94557632],
            [116.44461394,39.94002939],
            [116.44303804,39.93636882],
            [116.44298545,39.92775136],
            [116.44223875,39.93254083],
            [116.44085231,39.94319385],
            [116.44052333 , 39.95066111],
            [116.439516   , 39.92758311],
            [116.4390503  , 39.94006242],
            [116.43861083 , 39.9371425 ],
            [116.43833667 , 39.94766083],
            [116.43804088 , 39.93335676],
            [116.4360725  , 39.9601625 ],
            [116.43607175 , 39.93949429],
            [116.43551643 , 39.93642905],
            [116.433018   , 39.942302  ],
            [116.432696   , 39.95135533],
            [116.43258696 , 39.95896087],
            [116.43230524 , 39.95582619],
            [116.43160143 , 39.92525238],
            [116.431434   , 39.9294056 ],
            [116.42908297 , 39.93371108],
            [116.42779577 , 39.9405425 ],
            [116.42681684 , 39.92990395],
            [116.42663921 , 39.92539789],
            [116.42545686 , 39.95260771],
            [116.42497    , 39.94660437],
            [116.42492    , 39.96024526],
            [116.42481954 , 39.93512708],
            [116.42468744 , 39.95689179],
            [116.42270973 , 39.94158784],
            [116.42181667 , 39.96116444],
            [116.42107375 , 39.92422292],
            [116.41972333 , 39.95477111],
            [116.41884333 , 39.94918286],
            [116.41872485 , 39.95828273],
            [116.41795556 , 39.96917889],
            [116.41745517 , 39.93134707],
            [116.41735    , 39.9274214 ],
            [116.41704306 , 39.93729014],
            [116.41670706 , 39.964855  ],
            [116.41648396 , 39.94530292],
            [116.41607554 , 39.94135973],
            [116.41603957 , 39.92483872],
            [116.41412421 , 39.96159053],
            [116.4136125  , 39.96959812],
            [116.41269903 , 39.95869935],
            [116.41071    , 39.96958613],
            [116.41056968 , 39.92626839],
            [116.40952898 , 39.94813816],
            [116.40901167 , 39.934445  ],
            [116.409      , 39.974165  ],
            [116.40881355 , 39.94026452],
            [116.40880697  ,39.93118545],
            [116.408595    ,39.96598583],
            [116.40843778 , 39.94452984],
            [116.4084125   ,39.9555325 ],
            [116.40814206 , 39.96263088],
            [116.40772955,  39.971685  ],
            [116.40749034  ,39.96094138],
            [116.40685378 , 39.95183405],
            [116.40574878,  39.92433366],
            [116.40485143  ,39.95802929],
            [116.40480357 , 39.97327214],
            [116.40413261,  39.96317087],
            [116.40341933  ,39.94043078],
            [116.403381   , 39.96954   ],
            [116.40334545,  39.93508364],
            [116.4029278   ,39.93769508],
            [116.4020955  , 39.9671495 ],
            [116.40166714,  39.9253581 ],
            [116.400888    ,39.944915  ],
            [116.399914   , 39.95153967],
            [116.39943571,  39.96925   ],
            [116.39875571  ,39.95569714],
            [116.39864029 , 39.94071559],
            [116.39849062,  39.96316813],
            [116.39772 ,    39.929778  ],
            [116.39716176  ,39.96587   ],
            [116.3969163  , 39.93370413],
            [116.3962297 ,  39.93532212],
            [116.39508774  ,39.93915833],
            [116.39507364 , 39.94636818],
            [116.39497111,  39.96319963],
            [116.39414824  ,39.95096059],
            [116.39406194 , 39.95445306],
            [116.39346962,  39.95716846],
            [116.39311278  ,39.94142944],
            [116.39307111 , 39.92625222],
            [116.39252385 , 39.93457846],
            [116.39068   ,  39.94647   ],
            [116.390155   , 39.951741  ],
            [116.387785  ,  39.95452687],
            [116.386016   , 39.95202867],
            [116.38354     ,39.95135778],
            [116.38235917 , 39.95690083]])

#各用户需求点需求量
xqd_xq = [151480,162300,102790,178530,275910,119020,129840,70330,48690,243450,178530,194760,64920,183940,21640,340830,227220,162300,81150,124430,113610,113610,135250,200170,281320,205580,205580,189350,86560,102790,351650,210990,200170,48690,129840,48690,113610,178530,48690,313780,308370,389520,183940,259680,400340,254270,102790,86560,167710,167710,167710,265090,259680,64920,503130,178530,129840,340830,129840,183940,119020,156890,200170,221810,75740,75740,124430,486900,108200,238040,319190,108200,113610,54100,162300,75740,37870,183940,86560,27050,91970,248860,178530,454440,59510,146070,183940,194760,140660,97380,48690,70330,5410,54100,86560,81150,146070,124430]




c1 = 0.02 #配送中心至末端共配网点单位重量、单位距离快递货物的运输费用
c2 = 0.4 #末端共配网点至顾客群聚集点单位重量、单位距离快递货物的运输费用
c3 = 0.4 #顾客群聚集点到末端共配网点自提时单位重量、单位距离快递货物的运输费用[元/( )]
c4 = [0.008,0.007,0.008] #不同规模网点风险成本单价
S = 1 #配送中心数
M = 15 #备选末端共配网点个数
N = 98 #顾客群聚集点个数
L = 3 #末端共同配送网点的规模种类
#di =           #配送中心k到末端共配网点i的距离_bxd_jl
#dij =          #末端共配网点i到顾客群聚集点j的距离_distance[i][j]
aik = 1          #配送中心k到末端共配网点i的货物运量
vij = 1          #末端共配网点i到顾客群聚集点j的货物运量
delta_b = 0.05  #顾客聚集点j选择去末端共配网点i自提的货物量占该末端共配网点总货物处理量的比例
xi = 1           #表示是否选择i建设规模为v的末端共配网点；
yij = 1          #表示是否由规模为v的末端共配网点i为第j个顾客群聚集点配送货物；
h1min = 0       #规模为1的末端共配网点i配送货物最小服务量
h1max = 4000    #规模为1的末端共配网点i配送货物最大服务量
h2min = 4000    #规模为2的末端共配网点i配送货物最小服务量
h2max = 8000    #规模为2的末端共配网点i配送货物最大服务量
h3min = 8000    #规模为3的末端共配网点i配送货物最小服务量
h3max = 12000    #规模为3的末端共配网点i配送货物最大服务量
Mkmax = 999999  #配送中心k配送货物最大能力
A = 999999      #为一定时期（通常为1年）内末端共配企业在末端共配网点上的投资的最大预算
delta_n = 100   #末端共配网点运营成本与货物处理量的系数比

def leijia(a):
    q = 0
    for i in range(len(a)):
        q = q + a[i]
    return q

def list_mulmin(k,a):
    s = []
    for i in range(len(a)):
        s.append(k * (a[i]-1))
    return s

def list_mulmax(k,a):
    s = []
    for i in range(len(a)):
        s.append(k * a[i])
    return s

def list_cheng(a,b):
    c = []
    for i in range(len(a)):
        c.append(a[i] * b[i])
    return c

def list_jia(a):
    s = 0
    for i in range(len(a)):
        s = s + a[i]
    return s

def list_plus(a,b):
    s = 0
    for i in range(len(a)):
        s = s + a[i] * b[i]
    return s

def minlist(a):
    min = 99999
    for i in range(len(a)):
        if a[i]==0:
            continue
        elif a[i]<=min:
            min = a[i]
            index = i
    return min,index

# 求两个向量之间的距离
def distEclud(vecA, vecB):
    lng1 = vecA[0]
    lat1 = vecA[1]
    lng2 = vecB[0]
    lat2 = vecB[1]
    lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    dis = 2 * asin(sqrt(a)) * 6371 * 1000
    dis = round(dis / 1000, 5)
    return dis

def kefou(a,b,c):
    s = 1
    for i in range(len(a)):
        if a[i]<b[i]<=c[i]:
            continue
        else:
            s = 0
    return s

#############################################################################
#建立需求点到备选点距离矩阵distance[i][j],i表示第i个需求点，j表示第j个备选点
distance = numpy.zeros(shape=(98,15))
for i in range(len(xqd)):
    for j in range(len(bxd)):
        distance[i][j] = distEclud(xqd[i],bxd[j])
###############################################################################
#记录备选点选择情况列表
#xd = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
xd = [0, 0, 2, 2, 1, 1, 1, 1, 1, 3, 3, 3, 1, 2, 2]
bxd_min = list_mulmin(1460000,xd)
bxd_max = list_mulmax(1460000,xd)
#print(bxd_min)
#print(bxd_max)

#创建还原矩阵和原始矩阵
xd_hy = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
xd_ys = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
for i in range(15):
    xd_ys[i] = xd[i]
    if xd[i]!=0:
        xd_hy[i] = 1
#print(xd_hy)

#记录最短距离矩阵
for i in range(98):
    distance[i] = list_cheng(distance[i],xd_hy)

#记录各点
select = numpy.zeros(shape=(2,98))
for i in range(98):
    select[0][i] = minlist(distance[i])[0]
    select[1][i] = minlist(distance[i])[1]
print(select)

# 建立一个数组储存各网点配送的货物量
bxd_yl = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
for i in range(98):
    bxd_yl[int(select[1][i])] = bxd_yl[int(select[1][i])] + xqd_xq[i]
#print(bxd_yl)

#判断该方案是否可行
# if kefou(bxd_min,bxd_yl,bxd_max)==0:
#     print('该方案不可行')
#     sys.exit(0)
for i in range(15):
    if 912500 <= bxd_yl[i] < 1825000:
        xd[i] = 2
    elif 1825000 <= bxd_yl[i] <2737500:
        xd[i] = 3
    if 2737500  < bxd_yl[i] :
        print('该方案不可行')
        sys.exit(0)



# 建立一个矩阵储存各备选网点到各聚集点的货物量*距离
qxd_yj = numpy.zeros(shape=(1,98))
s_qxd = 0
for i in range(98):
    # if select[0][i] >= 1:
    #     qxd_yj[0][i] = select[0][i] * xqd_xq[i]
    # else:
        qxd_yj[0][i] = select[0][i] * xqd_xq[i] * 0.95
    #s_qxd = s_qxd + qxd_yj[0][i]
# print(qxd_yj)
# print(s_qxd)

# 网点建设成本
bxd_v = []
wv = [20, 28, 34, 0]  # 建设规模为v的末端共同配送网点的固定成本
for i in range(len(xd)):
    bxd_v.append(wv[xd[i] - 1])
# print(bxd_v)

# 网点运营成本
bxd_yy = []
for i in range(len(bxd_yl)):
    bxd_yy.append(sqrt(bxd_yl[i]) * 100)
# print(bxd_yy)

###########################################################################################
# 建立一个矩阵储存乘客取件距离成本,即乘客自取，需要各聚集点到各备选网点的货物量*距离（自取）
qxd_zq = numpy.zeros(shape=(1,98))
s_qxdzq = 0
for i in range(98):
    # if select[0][i] >= 0.25:
    #     qxd_zq[0][i] = select[0][i] * xqd_xq[i] * 0
    # else:
        qxd_zq[0][i] = select[0][i] * xqd_xq[i] * 0.05
    #s_qxdzq = s_qxdzq + qxd_zq[0][i]
# print(qxd_yj)
# print(s_qxd)

#计算乘客取件时间成本
qxd_sj = numpy.zeros(shape=(1,98))
s_qxd = 0
for i in range(98):
    if select[0][i]/10 <= 0.25:
        qxd_sj[0][i] = 0
    # elif select[0][i] >= 0.5 :
    #     qxd_sj[0][i] = ((8 * (select[0][i]/10 - 0.5))**2) * xqd_xq[i]
    else:
        qxd_sj[0][i] = ((8 * (select[0][i]/10 - 0.5))**2) * xqd_xq[i] * 0.95
#print(qxd_sj)

# 建立一个矩阵存储——用户风险成本系数
bxd_fx = []
for i in range(len(xd)):
    bxd_fx.append(c4[xd[i]-1])
#print(bxd_fx)
# 计算风险成本
qxd_fx = []
for i in range(98):
    qxd_fx.append(xqd_xq[i] * bxd_fx[int(select[1][i])])
#print(qxd_fx)



#企业成本最优
f1 = c1 * list_plus(bxd_yl , bxd_jl)
f11 = c2 * list_jia(qxd_yj[0])    # 配送运输成本
f2 = list_jia(bxd_v) * 10000# 网点建设成本
f3 = list_jia(bxd_yy)# 网点运营成本
F = f1 + f11 + f2 + f3
print('全年企业成本 = ',F,'日均企业成本 = ',F/365)
print('配送中心到末端网点运输成本 = ',f1)
print('配送中心到用户聚集点运输成本 = ',f11)
print('网点建设成本 = ',f2)
print('网店运营成本 = ',f3)
#用户成本最优
# k1 = 0.2
# k2 = 0.4
# k3 = 0.4
# aj = #顾客j可接受最长等待时间
# ftij = #时间等待成本
# qj = #为顾客群聚集点j在一定时期内的总货物需求量
# eM = 99999 #大的正数
t1 =  c3 * list_jia(qxd_zq[0]) # 乘客自取距离成本
t2 =  list_jia(qxd_sj[0])#乘客自取时间成本
t3 =  list_jia(qxd_fx)#乘客风险成本
T = t1+t2+t3
print('用户年成本 = ', T,'用户日均成本 = ',T/365)
print('用户年距离成本 = ', t1)
print('用户年时间成本 = ', t2)
print('用户年风险成本 = ', t3)
#
print('原始方案:',xd_ys)
print('还原方案:',xd_hy)
print('改进方案:',xd)

#颜色备选
color= ['black','red','blue','orange','brown','darkgreen','cyan','pink','gold','purple','green','gray','navy','darkred','deepskyblue']
#网点大小尺寸
size = ['v','d','*']

fig1 = plt.figure(num='备选点', figsize=(7,7), dpi=75)

plt.plot(116.4108311,39.92814972,size[0], color = 'black',linewidth=0.5, markersize=8,label='small')
plt.plot(116.4108311,39.92814972,size[1], color = 'black',linewidth=0.5, markersize=8,label='medium')
plt.plot(116.4108311,39.92814972,size[2], color = 'black',linewidth=0.5, markersize=8,label='large')
plt.plot(116.4108311,39.92814972,"o", color = 'black',linewidth=0.5, markersize=4,label='custom')
for i in range(len(xd)):
    if xd_hy[i] == 0:
        continue
    else:
        plt.plot(bxd[i, 0], bxd[i, 1], size[xd[i]-1], color = color[i],linewidth=0.5, markersize=15)

for i in range(98):
    plt.plot(xqd[i,0],xqd[i,1],"ro",color = color[int(select[1][i])],linewidth=0.5, markersize=5)
    plt.plot([xqd[i,0], bxd[int(select[1][i]),0]], [xqd[i,1], bxd[int(select[1][i]),1]], color=color[int(select[1][i])],linewidth=0.5)
# for i in range(len(bxd)):
#     plt.plot(bxd[i,0],bxd[i,1],"r*",linewidth=0.5, markersize=10)
# plt.xlabel('')
# plt.ylabel('ouput number')
# plt.title('ouput number')
plt.legend()
plt.show()



# 配送中心坐标
# center = [116.593293,39.767526]
# bxd_jsjl = []

# for i in range(15):
#     bxd_jsjl.append(distEclud(center,bxd[i]))
#
# print(bxd_jsjl)