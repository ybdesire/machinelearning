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
algo = SVD()
# 在数据集上测试一下效果
perf = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3)
#输出结果
print_perf(perf)