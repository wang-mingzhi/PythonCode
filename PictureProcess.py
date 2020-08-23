# -*- coding: utf-8 -*-
# CreatedTime: 2020/8/23 8:29
# Email: 1765471602@qq.com
# File: PictureProcess.py
# Software: PyCharm
# Describe:
import cv2
import os
import requests
import time
import imageio
from PIL import Image, ImageFont, ImageDraw


def getpicture():
    url = "http://restapi.amap.com/v3/staticmap?"
    params = {'location': '116.282104,39.915929', 'zoom': 14, 'size': '2048*2048',
              'scale': 2, 'traffic': 1, 'key': 'eff48ee434d763609e59839fa946b9e1'}
    r_text = requests.get(url, params=params)
    r_text.raise_for_status()  # 当出现错误时及时抛出错误
    filename = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time())) + ".png"
    with open("D:/拥堵区域变化趋势/" + filename, 'wb') as f:
        f.write(r_text.content)


def addtext2pic(imgpath):
    for root, dirs, files in os.walk(imgpath):
        for file in files:
            fullpath = os.path.join(root, file)
            if 'png' not in file:
                continue
            name = os.path.splitext(file)[0]
            img1 = Image.open(fullpath).convert('RGBA')  # 用RGBA的模式打开图片,
            img2 = Image.new('RGBA', img1.size, (0, 0, 0, 0))  # 创建一个和原图一样大小的透明图片

            fnt = ImageFont.truetype("c:/Windows/Fonts/simsun.ttc", 100)
            d = ImageDraw.Draw(img2)
            d.text(xy=(750, 50), text=name, font=fnt, fill=(0, 0, 0, 95))
            out = Image.alpha_composite(img1, img2)  # 合并两张图片,并确定水印的位置
            out = out.quantize(256)
            out.save('{0}\\{1}_new.{2}'.format(root, name, 'png'))
            print(name)
        print(root)


def pic2gif(filepath):
    for root, dirs, files in os.walk(filepath):
        images = []
        for file in files:
            if 'png' not in file:
                continue
            images.append(imageio.imread(os.path.join(root, file)))
        imageio.mimsave('{0}.gif'.format(filepath), images, 'GIF', duration=1, loop=1)


def pic2video(path, size):
    filelist = os.listdir(path)  # 获取该目录下的所有文件名
    fps = 1  # fps(帧率):1秒钟有n张图片写进去[控制一张图片停留5秒钟，那就是帧率为1，重复播放这张图片5次]
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')  # 不同视频编码对应不同视频格式（例：'I','4','2','0' 对应avi格式）
    video = cv2.VideoWriter('{0}.mp4'.format(path), fourcc, fps, size)
    for item in filelist:
        if item.endswith('.png'):  # 判断图片后缀是否是.png
            item = path + '/' + item
            img = cv2.imread(item)  # 使用opencv读取图像，直接返回numpy.ndarray对象，通道顺序为BGR，注意是BGR，通道值默认范围0-255。
            video.write(img)  # 把图片写进视频
    video.release()  # 释放


if __name__ == "__main__":
    for k in range(145):
        getpicture()
        print(k + 1, time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time())))
        time.sleep(600)  # 休息600秒，也即是每隔10分钟获取一次图片
    addtext2pic(r"D:\拥堵区域变化趋势")
    pic2gif(r"D:\202008\20200823")
    pic2video(r"D:\202008\20200821", (2048, 2048))
    print('Done!')
