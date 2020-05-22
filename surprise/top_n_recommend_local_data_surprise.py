from collections import defaultdict

from surprise import SVD
from surprise import Dataset,Reader
import os


def get_top_n(predictions, n=10):

    # First map the predictions to each user.
    top_n = defaultdict(list)
    # uid： 用户ID
    # iid： item ID
    # true_r： 真实得分
    # est：估计得分
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    # 为每一个用户都寻找K个得分最高的item
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


# 指定文件所在路径
file_path = os.path.expanduser('mydata.csv')
# 告诉文本阅读器，文本的格式是怎么样的
reader = Reader(line_format='user item rating', sep=',')
# 加载数据
data = Dataset.load_from_file(file_path, reader=reader)
trainset = data.build_full_trainset()

algo = SVD()
algo.fit(trainset)

testset =  [
    ('5','1',0),# 想获取第5个用户对第1个item的得分
    ('5','4',0),# 0这个位置是真实得分，不知道时可以写0
    ('5','5',0),# 但写0后，就没法进行算法评估了，因为不知道真实值
]
predictions = algo.test(testset)

top_n = get_top_n(predictions, n=2)

# Print the recommended items for each user
for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])