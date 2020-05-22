from surprise import SVD
from surprise import Dataset, print_perf, Reader
from surprise.model_selection import cross_validate
import os

# 指定文件所在路径
file_path = os.path.expanduser('mydata.csv')
# 告诉文本阅读器，文本的格式是怎么样的
reader = Reader(line_format='user item rating', sep=',')
# 加载数据
data = Dataset.load_from_file(file_path, reader=reader)
trainset = data.build_full_trainset()

algo = SVD()
algo.fit(trainset)

# we can now query for specific predicions
uid = str(5)  # raw user id
iid = str(1)  # raw item id

# get a prediction for specific users and items.
pred = algo.predict(uid, iid)
print('rating of user-{0} to item-{1} is '.format(uid, iid), pred.est)# rating of user-5 to item-1

#----------------------------
uid = str(5)  # raw user id
iid = str(5)  # raw item id
# get a prediction for specific users and items.
pred = algo.predict(uid, iid)
print('rating of user-{0} to item-{1} is '.format(uid, iid), pred.est)
