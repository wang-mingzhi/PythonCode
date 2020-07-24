# -*- coding: utf-8 -*-
# CreatedTime: 2020/7/24 8:39
# Email: 1765471602@qq.com
# File: testfile.py
# Software: PyCharm
# Describe:
import time


@profile
def fun():
    a = 0
    b = 0
    for i in range(100000):
        a = a + i * i
    for i in range(3):
        b += 1
        time.sleep(0.1)
    return a + b


if __name__ == "__main__":
    fun()
