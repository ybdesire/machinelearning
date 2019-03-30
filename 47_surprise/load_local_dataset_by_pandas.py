import numpy as np
import pandas as pd
header = ['user_id','item_id','rating','timestamp']
src_data = pd.read_csv('ml-100k/u.data',sep = '\t',names = header)

unique_user_count = src_data.user_id.nunique()                    #查看用户去重个数
unique_item_count = src_data.item_id.nunique()                    #查看物品去重个数


print('unique_user_count=', unique_user_count)
print('unique_item_count=', unique_item_count)

