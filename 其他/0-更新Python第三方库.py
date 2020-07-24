# -*- coding: utf-8 -*-
"""
Author: 18120900
Created: 2020/5/19 19:27
Software: PyCharm
Desc: 更新python第三方库
"""

import subprocess
import progressbar


def getoutdatedlib():
    command = 'pip list --outdated'
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    print('正在查找需要更新的第三方库！这一过程可能需要一段时间...')
    outdatedlib = [line.strip().decode() for line in p.stdout.readlines()]
    print('\r\n'.join(outdatedlib))
    return outdatedlib[2:]


def updatelib(outdatedlib):
    bar = progressbar.ProgressBar(max_value=len(outdatedlib))
    errorlist = []
    for i, dist in enumerate(outdatedlib):
        libname = dist.split(' ')[0]
        if libname == 'pip':
            errorlist.append(dist)
            print('pip需要单独更新！')
        else:
            command = 'pip install --upgrade ' + libname
            status = subprocess.call(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            if status != 0:
                errorlist.append(dist)
            else:
                print(libname, '\t更新完成。')
        bar.update(i+1)
    return errorlist


if __name__ == '__main__':
    outdatedlib = getoutdatedlib()
    errorlist = updatelib(outdatedlib)
    print('\r\n共{}个库！{}个库更新失败'.format(len(outdatedlib), len(errorlist)))
    if len(errorlist) > 0:
        print('\r\n'.join(errorlist))
