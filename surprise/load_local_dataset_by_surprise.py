import os
from surprise import Dataset, Reader

# 指定文件所在路径
file_path = os.path.expanduser('ml-100k/u.data')
# 告诉文本阅读器，文本的格式是怎么样的
reader = Reader(line_format='user item rating timestamp', sep='\t')
# 加载数据
data2 = Dataset.load_from_file(file_path, reader=reader)
trainset = data2.build_full_trainset()# 转换成这种结构，才能获取到数据集详细信息
print('user count: ', trainset.n_users)# 用户数
print('item count', trainset.n_items)#电影数