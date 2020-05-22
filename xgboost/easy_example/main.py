import xgboost as xgb
import numpy as np

x_train = np.asarray([[0,0,0],[1,1,1],[2,2,2]])
y_train = np.asarray([0,1,2])

x_valid = np.asarray([[0.2,0.1,0.2],[1.1,1.2,1.3],[2.1,2.2,2.3]])
y_valid = np.asarray([0,1,2])

x_test = np.asarray([[0.1,0.2,0.3],[0.4,0.1,0.3],[1,1.5,1],[1.9,2,1.8]])


params={
'booster':'gbtree',
'objective': 'multi:softmax', 
'num_class':3, # 类数，与 multisoftmax 并用
'gamma':0.05,  # 在树的叶子节点下一个分区的最小损失，越大算法模型越保守 。[0:]
'max_depth':12, # 构建树的深度 [1:]
#'lambda':450,  # L2 正则项权重
'subsample':0.4, # 采样训练数据，设置为0.5，随机选择一般的数据实例 (0:1]
'colsample_bytree':0.7, # 构建树树时的采样比率 (0:1]
#'min_child_weight':12, # 节点的最少特征数
'silent':1 ,
'eta': 0.005, # 如同学习率
'seed':710,
'nthread':4,# cpu 线程数
}

plst = list(params.items())

xgtrain = xgb.DMatrix(x_train, y_train)
xgval = xgb.DMatrix(x_valid, y_valid)
watchlist = [(xgtrain, 'train'),(xgval, 'val')]
xgtest = xgb.DMatrix(x_test)

num_rounds = 500

model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=100)
pred = model.predict(xgtest,ntree_limit=model.best_iteration)

print(pred)