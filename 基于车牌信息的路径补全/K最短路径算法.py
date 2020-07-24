# -*- coding: utf-8 -*- 
"""
Author: 18120900
Created: 2020/7/8 19:08
Software: PyCharm
Desc: 
"""
import pandas as pd
import heapq
import sys
import copy
import datetime
import os


class Graph:
    def __init__(self):
        self.vertices = {}

    def add_vertex(self, name, edges):
        self.vertices[name] = edges

    def get_shortest_path(self, startpoint, endpoint):
        """
        @param startpoint: 起始点
        @param endpoint: 终点
        @return: distances：每一个顶点到startpoint点的距离
                 shortest_path：最短路径信息，是一系列点的集合
                 lenPath：最短路径的长度
        """

        # distances使用字典的方式保存每一个顶点到startpoint点的距离
        distances, shortest_path, lenPath = {}, [], []

        # 从startpoint到某点的最优路径的前一个结点
        # eg:startpoint->B->D->E,则previous[E]=D,previous[D]=B,等等
        previous = {}

        # 用来保存图中所有顶点的到startpoint点的距离的优先队列, [distance, pointname]
        nodes = []
        # Dikstra算法 数据初始化
        for vertex in self.vertices:
            if vertex == startpoint:
                # 将startpoint点的距离初始化为0
                distances[vertex] = 0
                heapq.heappush(nodes, [0, vertex])
            elif vertex in self.vertices[startpoint]:
                # 把与startpoint点相连的结点距离startpoint点的距离初始化为对应的弧长/路权
                distances[vertex] = self.vertices[startpoint][vertex]
                heapq.heappush(nodes, [self.vertices[startpoint][vertex], vertex])
                previous[vertex] = startpoint
            else:
                # 把与startpoint点不直接连接的结点距离startpoint的距离初始化为sys.maxsize
                distances[vertex] = sys.maxsize
                heapq.heappush(nodes, [sys.maxsize, vertex])
                previous[vertex] = None

        while nodes:
            # 取出队列中最小距离的结点
            smallest = heapq.heappop(nodes)[1]
            if smallest == endpoint:
                shortest_path = []
                lenPath = distances[smallest]
                temp = smallest
                while temp != startpoint:
                    shortest_path.append(temp)
                    temp = previous[temp]
                # 将startpoint点也加入到shortest_path中
                shortest_path.append(temp)
            if distances[smallest] == sys.maxsize:
                # 所有点不可达
                break
            # 遍历与smallest相连的结点，更新其与结点的距离、前继节点
            for neighbor in self.vertices[smallest]:
                dis = distances[smallest] + self.vertices[smallest][neighbor]
                if dis < distances[neighbor]:
                    distances[neighbor] = dis
                    # 更新与smallest相连的结点的前继节点
                    previous[neighbor] = smallest
                    for node in nodes:
                        if node[1] == neighbor:
                            # 更新与smallest相连的结点到startpoint的距离
                            node[0] = dis
                            break
                    heapq.heapify(nodes)
        return distances, shortest_path, lenPath

    def getMinDistancesIncrement(self, inputlist):
        lenList = [v[0] for v in inputlist]
        minValue = min(lenList)
        minValue_index = lenList.index(minValue)
        minPath = [v[1] for v in inputlist][minValue_index]
        return minValue, minPath, minValue_index

    def deleteCirclesWithEndpoint(self, inputlist, endpoint):
        """
        该函数主要是删除类似于这样的例子： endpoint->...->endpoint-->...
        """
        pathsList = [v[1] for v in inputlist]
        for index, path in enumerate(pathsList):
            if len(path) > 1 and path[-1] == endpoint:
                inputlist.pop(index)
        return inputlist

    def k_shortest_paths(self, start, end, k=3):
        """
        :param start: 起始点
        :param end: 终点
        :param k: 给出需要求的最短路数
        :return: 返回K最短路和最短路长度
        该算法重复计算了最短路，调用get_shortest_path()方法只是用到了起始点到其他所有点的最短距离和最短路长度
        """
        distances, _, shortestPathLen = self.get_shortest_path(start, end)
        num_shortest_path = 0
        # paths: key=path, value=path_distance
        paths = dict()
        distancesIncrementList = [[0, [end]]]
        while num_shortest_path < k:
            # distancesIncrementList = self.deleteCirclesWithEndpoint(distancesIncrementList, end)
            minValue, minPath, minIndex = self.getMinDistancesIncrement(distancesIncrementList)
            smallest_vertex = minPath[-1]
            distancesIncrementList.pop(minIndex)

            if smallest_vertex == start:
                num_shortest_path += 1
                paths[tuple(minPath[::-1])] = minValue + shortestPathLen
                continue

            for neighbor in self.vertices[smallest_vertex]:
                incrementValue = copy.deepcopy(minPath)
                increment = 0
                if neighbor == end:
                    # 和函数deleteCirclesWithEndpoint()作用一样
                    continue
                if distances[smallest_vertex] == (distances[neighbor] + self.vertices[smallest_vertex][neighbor]):
                    increment = minValue
                elif distances[smallest_vertex] < (distances[neighbor] + self.vertices[smallest_vertex][neighbor]):
                    increment = minValue + distances[neighbor] + self.vertices[smallest_vertex][neighbor] - distances[
                        smallest_vertex]
                elif distances[neighbor] == (distances[smallest_vertex] + self.vertices[smallest_vertex][neighbor]):
                    increment = minValue + 2 * self.vertices[smallest_vertex][neighbor]
                incrementValue.append(neighbor)
                distancesIncrementList.append([increment, incrementValue])
        return paths


def creatGraph(sheet):
    vertexs = {}
    sheet['P1'] = sheet['P1'].astype('str')
    sheet['P2'] = sheet['P2'].astype('str')
    for i in range(len(sheet)):
        if sheet.iloc[i, 0] not in vertexs:
            vertexs[sheet.iloc[i, 0]] = {sheet.iloc[i, 1]: sheet.iloc[i, 2]}
        else:
            vertexs[sheet.iloc[i, 0]].update({sheet.iloc[i, 1]: sheet.iloc[i, 2]})

    g = Graph()
    for key, value in vertexs.items():
        g.add_vertex(key, value)
    return g, vertexs


def pathComplete(data, vertexs, graph, maxtimespan):
    result = []
    route_dict = {}
    # 设置两相邻记录的最大时间差
    data['code'] = data['code'].astype('str')
    data['datetime'] = pd.to_datetime(data['datetime'])
    # 根据时间字段对group进行升序排列
    data = data.sort_values(by=['datetime', 'platetype', 'platenumber'], ascending=True)
    grouped_data = data.groupby(['platenumber', 'platetype'])
    count = 0
    for name, group in grouped_data:
        count += 1
        if count % 1000 == 0 or count == 1:
            print('共有{}组数据,已处理{}组数据'.format(len(grouped_data), count))
        # 判断同一车牌号、同一类型车辆的记录数是否小2条，若满足条件则忽略该组
        if len(group) < 2:
            continue

        # 遍历group中的每条记录
        for i in range(1, len(group)):
            # 如果相邻记录的时间差大于maxtimespan则认为是冗余点；或：若相邻两条记录的点id相同，则肯定不存在缺失点
            if group.iloc[i, 2] - group.iloc[i - 1, 2] > maxtimespan or group.iloc[i, 3] == group.iloc[i - 1, 3]:
                continue
            # 判断相邻记录中的点id是否在路网中存在
            if group.iloc[i, 3] not in vertexs or group.iloc[i - 1, 3] not in vertexs:
                print('点{}或{}在路网上未找到！！！'.format(group.iloc[i, 3], group.iloc[i - 1, 3]))
                continue
            # 判断相邻记录中的点id是否直接相邻
            if vertexs[group.iloc[i, 3]].get(group.iloc[i - 1, 3], 0) > 0:
                continue
            if vertexs[group.iloc[i - 1, 3]].get(group.iloc[i, 3], 0) > 0:
                continue

            # 设置需要补全路径的起点：start，终点：end，以及查找结果数：k
            start, end, k = group.iloc[i - 1, 3], group.iloc[i, 3], 4
            index = 1
            start, end = (start, end) if start <= end else (end, start)
            if (start, end) not in route_dict.keys():
                route_dict[(start, end)] = {}
                for path, length in graph.k_shortest_paths(start, end, k).items():
                    value = [index, start, end, ",".join(path), length]
                    route_dict[(start, end)][index] = value
                    index += 1

            for k, v in route_dict[(start, end)].items():
                temp = [group.iloc[i - 1, 4], group.iloc[i, 4]]
                temp.extend(v)
                result.append(temp)
    return result


def main():
    if not os.access('result.csv', os.W_OK):
        print('请先关闭文件：result.csv !!!')

    sheets = pd.read_excel('计算用表.xlsx', sheet_name=[0, 1])
    g, vertexs = creatGraph(sheets[1])  # 创建路网

    maxtimespan = datetime.timedelta(hours=0, minutes=30, seconds=0)
    result = pathComplete(sheets[0], vertexs, g, maxtimespan)
    df = pd.DataFrame(columns=['起点id', '终点id', '序号', '起点', '终点', '路径', '长度/m'], data=result)
    df.to_csv('result.csv', index=False, encoding='utf_8_sig')
    print('Done')

    # start, end, k = '1', '5', 4
    # for path, length in g.k_shortest_paths(start, end, k).items():
    #     print(start, end, ",".join(path), length)


if __name__ == '__main__':
    main()
