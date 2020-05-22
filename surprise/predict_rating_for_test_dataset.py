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

# 训练SVD算法
algo = SVD()
algo.fit(trainset)

testset = [
    ('5','1',1),# 想获取第5个用户对第1个item的得分
    ('5','4',4),# 最后一位是真实得分
    ('5','5',5),# 最后一位也可以写0，这会导致计算RMSE等评价指标时不准确，但对预测值没有影响
]

# Than predict ratings for all pairs (u, i) that are NOT in the training set.
predictions = algo.test(testset)

# est is prediction rating
print(predictions)

