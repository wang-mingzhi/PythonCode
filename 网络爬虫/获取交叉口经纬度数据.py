# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 10:37:29 2020
project name:get crossRoadName's longitude and laititude from 高德地图
@author: 18120900
"""
import pandas as pd
import requests
import json
import time


def getintersectionlatlng(temp_crossroadname, result, error):
    try:
        url_1 = 'https://restapi.amap.com/v3/geocode/geo?address='
        url_2 = '&batch=true&output=json&key=eff48ee434d763609e59839fa946b9e1'
        url = url_1 + '|'.join(temp_crossroadname) + url_2  # 对把交叉口名包含在url中

        r_text = requests.get(url)
        r_text.raise_for_status()  # 当出现错误时及时抛出错误
        content = json.loads(r_text.content)
        r_text.close()  # 很重要的一步！！！，否则会导致错误

        status = content["status"]
        for k in range(int(content["count"])):
            if status == "1":
                adcode = content["geocodes"][k]["adcode"]
                formatted_address = content["geocodes"][k]["formatted_address"]
                location = content["geocodes"][k]["location"]
                level = content["geocodes"][k]["level"]
                result.append((temp_crossroadname[k], formatted_address, adcode, location, level))
            else:
                error.append(temp_crossroadname[k])
                print('error!')
    except TimeoutError:
        print('timeout error')


if __name__ == "__main__":
    result = []  # 设置一个列表用来存放提取结果
    error = []  # 设置一个列表用来存放请求失败的交叉口数据
    result.append(('Name', 'formatted_address', 'adcode', 'location', 'level'))
    error.append('Name')

    with open(r'F:\18120900\桌面\地理逆编码.txt', 'r', encoding='utf-8') as f:
        crossRoad = f.readlines()
    print(len(crossRoad))

    temp_crossRoadName = []  # 设置一个列表用来存放交叉口名称
    i = -1
    for crossRoadName in crossRoad:
        i += 1
        temp_crossRoadName.append(crossRoadName.replace('\n', ''))
        if i % 10 == 9 or i == len(crossRoad) - 1:
            print(i + 1)  # 显示处理到那一个交叉口了
            getintersectionlatlng(temp_crossRoadName, result, error)
            temp_crossRoadName.clear()
            time.sleep(2)
    df = pd.DataFrame(result)
    df.to_excel(r'F:\18120900\桌面\地理逆编码处理结果.xlsx')
    print('Finished')
