from surprise import Dataset

# 默认载入movielens数据集
data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()# 转换成这种结构，才能获取到数据集详细信息
print('user count: ', trainset.n_users)# 用户数
print('item count', trainset.n_items)#电影数