# -*- coding: utf-8 -*-
"""
Created on Tue May 19 17:34:31 2020

@author: ASUS
"""

for snum in range(25, 60):
    def factorials(x):
        if x == 0 or x == 1:
            return 1
        else:
            return x * factorials(x - 1)


class QueuingTheory(object):
    def __init__(self, ar, sr, snum, N):
        # ar:顾客到达率
        # sr:机构服务率
        # snum:充电车位数量
        # N:能容纳的最大车辆数
        self.ar = ar
        self.sr = sr
        self.snum = snum
        self.N = N
        self.ro = self.ar / self.sr
        self.ros = self.ar / (self.sr * self.snum)
        self.p0 = self.p0_Compute()  # 系统中所有充电车位空闲的概率
        self.cw = self.CW_Compute()  # 系统中车辆排队的概率
        self.lq = self.Lq_Compute()  # 系统中排队等待的平均车辆数
        self.ls = self.lq + self.ro  # 系统中的平均停留车辆数
        self.rw = self.ro / self.snum  # 系统中充电车位的平均利用率
        self.ws = self.ls / self.ar  # 系统中车辆的平均等待时间
        self.wq = self.lq / self.ar  # 系统中车辆的平均排队时间
        self.pf = self.Pf_Compute()

    def p0_Compute(self):  # 系统中所有充电桩空闲的概率
        result = 0
        ro, ros = self.ar / self.sr, self.ar / (self.sr * self.snum)
        for k in range(self.snum):
            result += ro ** k / factorials(k)
            result += ro ** self.snum / factorials(self.snum) / (1 - ros)
        return 1 / result if (1 / result) > 0 else 0

    def CW_Compute(self):  # 订单排队的概率
        ro, ros = self.ar / self.sr, self.ar / (self.sr * self.snum)
        return ro ** self.snum * self.p0 / factorials(self.snum) / (1 - ros)

    def Lq_Compute(self):  # 排队等待的平均车辆数
        ros = self.ar / (self.sr * self.snum)
        return self.cw * ros / (1 - ros)

    def Pf_Compute(self):  # 不再接纳顾客的概率
        return (self.ar / self.sr) ** snum / (factorials(self.snum) * self.snum ** (self.N - self.snum)) * self.p0


def main():
    inputresult = input('请输入系统到达率，服务率，充电车位数量，能容纳的最大车位数：\r\n')
    ar, sr, snum, N = map(eval, list(inputresult.split(',')))
    myqueuing = QueuingTheory(ar, sr, snum, N)

    def Per_Energy(self):  # 某一时间段内停车场充电负荷
        Q = [30, 30, 40, 45, 50] * 5  # Q:电动汽车电池容量
        energy = 0.8 * Q  # 每一辆车所需充的电量
        number = (1 - self.pf) * len(Q)  # number:实际充电的车辆数
        for i in range(1, int(number) + 1):
            energy += energy
            return (energy)

    print('系统中订单排队的概率为%6.3f' % myqueuing.cw)
    print('系统中排队等待的平均车辆数为%6.3f' % myqueuing.lq)
    print('系统中车辆的平均排队时间为%6.3f分钟' % (myqueuing.wq * 60))
    print('系统中车辆的平均等待时间为%6.3f分钟' % (myqueuing.ws * 60))
    print('系统总成本为%6.3f元' % (200 * snum + myqueuing.lq * 10))
    # 'QueuingTheory' object has no attribute 'energy'
    # print('系统效益为%6.3f元' % (1.2 * myqueuing.energy - 200 * snum - myqueuing.lq * 10))


if __name__ == '__main__':
    main()
