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
    filename = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
    with open('D:/{0}.{1}'.format(filename, 'jpg'), 'wb') as f:
        f.write(r_text.content)


def addtext2pic(imgpath):
    for file in os.listdir(imgpath):
        fullpath = os.path.join(imgpath, file)
        name = os.path.splitext(file)[0]
        fnt = ImageFont.truetype("c:/Windows/Fonts/simsun.ttc", 100)
        if file.endswith('png'):
            img1 = Image.open(fullpath).convert('RGBA')  # 用RGBA的模式打开图片,
            img2 = Image.new('RGBA', img1.size, (0, 0, 0, 0))  # 创建一个和原图一样大小的透明图片
            d = ImageDraw.Draw(img2)
            d.text(xy=(750, 50), text=name, font=fnt, fill=(0, 0, 0, 95))
            out = Image.alpha_composite(img1, img2)  # 合并两张图片,并确定水印的位置
            out = out.quantize(256)  # 压缩图片大小，大概能压缩75%
            out.save('{0}\\{1}_new.{2}'.format(imgpath, name, 'png'))
        elif file.endswith('jpg'):
            img1 = Image.open(fullpath)
            draw = ImageDraw.Draw(img1)
            draw.text(xy=(750, 50), text=name, font=fnt, fill=(0, 0, 0, 95))
            img1.save('{0}\\{1}_new.{2}'.format(imgpath, name, 'jpg'), format='jpg', quality=75)
        print(name)


def pic2gif(path):
    images = []
    for file in os.listdir(path):
        if file.endswith('png'):
            images.append(imageio.imread(path + '/' + file))
    imageio.mimsave('{0}.gif'.format(path), images, 'GIF', duration=1, loop=1)


def pic2video(path, size):
    fps = 1  # fps(帧率):1秒钟有n张图片写进去[控制一张图片停留5秒钟，那就是帧率为1，重复播放这张图片5次]
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')  # 不同视频编码对应不同视频格式（例：'I','4','2','0' 对应avi格式）
    video = cv2.VideoWriter('{0}.mp4'.format(path), fourcc, fps, size)
    for item in os.listdir(path):
        if item.endswith('.png'):  # 判断图片后缀是否是.png
            item = os.path.join(path, item)
            img = cv2.imread(item)  # 使用opencv读取图像，直接返回numpy.ndarray对象，通道顺序为BGR，注意是BGR，通道值0-255。
            video.write(img)  # 把图片写进视频
    video.release()  # 释放


if __name__ == "__main__":
    # for k in range(145):
    #     getpicture()
    #     print(k + 1, time.time())
    #     time.sleep(600)  # 休息600秒，即每隔10分钟获取一次图片
    # addtext2pic(r"D:\20200824")
    # pic2gif(r"D:\202008\20200823")
    pic2video(r"D:\202008", (2048, 2048))
    print('Done!')
