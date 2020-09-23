# -*- coding: utf-8 -*-
# CreatedTime: 2020/9/9 16:33
# Email: 1765471602@qq.com
# File: 视频文件处理.py
# Software: PyCharm
# Describe: 

import os


def savevideo(fileinputpath, fileoutputpath, size='1366x768'):
    """
    视频越长视频压缩时间越长（视CPU而定），需耐心等待
    @param fileinputpath: 输入文件路径
    @param fileoutputpath: 输出文件路径
    @param size: 视频的分辨率
    @return:
    """
    if os.path.getsize(fileinputpath) / 1024 < 150.0:  # 文件大于150k才开始处理
        print('{0}文件小于150kb，不需要进行处理'.format(fileinputpath))
        return
    """
    # -i 输入文件
    # -s 设置输出文件的分辨率,wxh；
    # -r 每一秒的帧数,一秒10帧大概就是人眼的速度
    # -i 输入的视频文件
    # -pix_fmt 设置视频颜色空间 yuv420p网络传输用的颜色空间 ffmpeg -pix_fmts可以查看有哪些颜色空间选择
    # -vcodec 软件编码器，libx264通用稳定
    # -preset 编码机预设 编码机预设越高占用CPU越大 有十个参数可选 ultrafast superfast veryfast(录制视频选用) 
    faster fast medium(默认) slow slower veryslow(压制视频时一般选用) pacebo
    # -profile:v 压缩比的配置 越往左边压缩的越厉害，体积越小 baseline(实时通信领域一般选用，画面损失越大) Extended Main(流媒体选用) 
    High(超清视频) High 10 High 4:2:2 High 4:4:4(Predictive)
    # -level:v 对编码机的规范和限制针对不通的使用场景来操作,也就是不同分辨率设置不同的值(这个要根据不同的分辨率进行设置的,具体要去官方文档查看)
    # -crf 码率控制模式 用于对画面有要求，对文件大小无关紧要的场景 0-51都可以选择 0为无损 一般设置18 - 28之间 大于28画面损失严重
    # -acodec 设置音频编码器
    """
    compress = "ffmpeg -i {0} -s {1} -vcodec libx265 -crf 20 {2}".format(fileinputpath, size, fileoutputpath)
    isRun = os.system(compress)
    return isRun, "没有安装ffmpeg" if isRun != 0 else True


def main():
    fileinputpath = r"C:\Users\王明智\Video\张靓颖-我的梦.mp4"
    fileoutputpath = r"C:\Users\王明智\Video\张靓颖-我的梦-Commpressed.mp4"
    savevideo(fileinputpath, fileoutputpath)


if __name__ == "__main__":
    # 唉，QQ影音处理视频更快、更好
    main()
    print('Done')
