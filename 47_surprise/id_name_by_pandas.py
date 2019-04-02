import numpy as np
import pandas as pd

def read_item_names():
    """
    获取电影名到电影id 和 电影id到电影名的映射
    """
    file_name = ('ml-100k/u.item')
    rid_to_name = {}
    name_to_rid = {}
    with open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split('|')
            rid_to_name[line[0]] = line[1]
            name_to_rid[line[1]] = line[0]
    return rid_to_name, name_to_rid
    
    
rid_to_name, name_to_rid = read_item_names()

print(rid_to_name['242'])# 242号电影的名字
print(name_to_rid['L.A. Confidential (1997)'])#'L.A. Confidential (1997)'电影的id