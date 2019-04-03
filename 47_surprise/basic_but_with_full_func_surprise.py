from surprise import SVD
from surprise import Dataset, print_perf
from surprise.model_selection import cross_validate

# 默认载入movielens数据集
data = Dataset.load_builtin('ml-100k')
algo = SVD()
# 在数据集上测试一下效果
perf = cross_validate(algo, data, measures=['RMSE'], cv=3)# RMSE（均方根误差）
#输出结果
print_perf(perf)