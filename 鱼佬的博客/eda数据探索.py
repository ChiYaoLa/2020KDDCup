# -*- coding: utf-8 -*-
# @Time    : 2020/6/30 12:07
"""https://zhuanlan.zhihu.com/p/127336206"""

### 2.1数据加载
# %matplotlib inline
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font', family='SimHei', size=13)

import os,gc,re,warnings,sys
warnings.filterwarnings("ignore")

path = './data/'

##### train
train_user_df = pd.read_csv(path+'underexpose_train/underexpose_user_feat.csv', names=['user_id','user_age_level','user_gender','user_city_level'])
train_item_df = pd.read_csv(path+'underexpose_train/underexpose_item_feat.csv')
train_click_0_df = pd.read_csv(path+'underexpose_train/underexpose_train_click-0.csv',names=['user_id','item_id','time'])

##### test
test_qtime_0_df = pd.read_csv(path+'underexpose_test/underexpose_test_click-0/underexpose_test_qtime-0.csv', names=['user_id','query_time'])
test_click_0_df = pd.read_csv(path+'underexpose_test/underexpose_test_click-0/underexpose_test_click-0.csv', names=['user_id','item_id','time'])


## 2.2 预处理
train_item_df.columns = ['item_id'] + ['txt_vec'+str(i) for i in range(128)] + ['img_vec'+str(i) for i in range(128)]
train_item_df['txt_vec0'] = train_item_df['txt_vec0'].apply(lambda x:float(x[1:]))
train_item_df['txt_vec127'] = train_item_df['txt_vec127'].apply(lambda x:float(x[:-1]))
train_item_df['img_vec0'] = train_item_df['img_vec0'].apply(lambda x:float(x[1:]))
train_item_df['img_vec127'] = train_item_df['img_vec127'].apply(lambda x:float(x[:-1]))

# rank 用户点击时间排名
train_click_0_df['rank'] = train_click_0_df.groupby(['user_id'])['time'].rank(ascending=False).astype(int)
test_click_0_df['rank'] = test_click_0_df.groupby(['user_id'])['time'].rank(ascending=False).astype(int)

# click cnts  用户点击次数
train_click_0_df['click_cnts'] = train_click_0_df.groupby(['user_id'])['time'].transform('count')
test_click_0_df['click_cnts'] = test_click_0_df.groupby(['user_id'])['time'].transform('count')


# 2.3 基本探查
train_user_df[:5]  #用户属性数据
train_user_df.info()

plt.figure()
plt.figure(figsize=(16, 10))
i = 1
for col in ['user_id', 'user_age_level', 'user_gender', 'user_city_level']:
    plt.subplot(2, 2, i)
    i += 1
    v = train_user_df[col].value_counts().reset_index()[:10] # 直接value_counts() 方便画柱状图
    fig = sns.barplot(x=v['index'], y=v[col])
    for item in fig.get_xticklabels():
        item.set_rotation(90)
    plt.title(col)
plt.tight_layout()
plt.show()

## 是否有用户重复点击
train_click_0_df.groupby(["user_id","item_id"])["time"].agg({"count"}).reset_index()

##  商品共现频次 着重看mean min max三个值，看看共现对的count分布
tmp = train_click_0_df.sort_values('time')
tmp['next_item'] = tmp.groupby(['user_id'])['item_id'].transform(lambda x:x.shift(-1)) # 创造每个人的next_item
union_item = tmp.groupby(['item_id','next_item'])['time'].agg({'count'}).reset_index().sort_values('count', ascending=False)
union_item[['count']].describe()

## 分析用户的点击序列 相邻行记录的关联性
tmp = train_click_0_df[train_click_0_df["user_id"]==5701]
tmp = tmp.merge(train_item_df,on="item_id",how="left")
tmp[tmp["txt_vec0"].isnull()] # 查看文本txt_vec0为null的相关行，看看那些位置空了，有啥影响

nonull_tmp = tmp[~tmp['txt_vec0'].isnull()] #非空

sim_list = []
for i in range(0, nonull_tmp.shape[0]-1):
    emb1 = nonull_tmp.values[i][-128-128:-128]    # 前一个的 txt_vec 特征
    emb2 = nonull_tmp.values[i+1][-128-128:-128]  # 后一个的 txt_vec 特征
    sim_list.append(np.dot(emb1,emb2)/(np.linalg.norm(emb1)*(np.linalg.norm(emb2)))) # 向量余弦值（相似度）
sim_list.append(0)

plt.figure()
plt.figure(figsize=(10, 6))
fig = sns.lineplot(x=[i for i in range(len(sim_list))], y=sim_list)
for item in fig.get_xticklabels():
    item.set_rotation(90)
plt.tight_layout()
plt.title('用户点击序列前后txt相似性')
plt.show()
# 同理也可以分析 图像等模态前后点击记录的相似性


"""接下来看看商品嵌入表示，是不是如同txt和img向量一样。这里使用word2vec进行构造，
当然还可以尝试图嵌入等方式来提取嵌入表示。"""
# 根据用户点击序列提取商品嵌入表示
tmp = train_click_0_df.sort_values('time')

# 一行代码提取用户点击序列，并构成文本
doc = tmp.groupby(['user_id'])['item_id'].agg({list}).reset_index()['list'].values.tolist()


from gensim.models import Word2Vec # 导入 Word2Vec


for i in range(len(doc)):
    doc[i] = [str(x) for x in doc[i]]   # 转为字符串型才能进行训练

model = Word2Vec(doc, size=128, window=5, min_count=3, sg=0, hs=1, seed=2020)


values = set(tmp['item_id'].values)
w2v = []

for v in values:
    try:
        a = [int(v)]
        a.extend(model[str(v)])  # int(v)是原始item_id向量，model[str(v)]提取出item_id的嵌入向量
        w2v.append(a)
    except:
        pass

out_df = pd.DataFrame(w2v)
out_df.columns = ['item_id'] + ['item_vec' + str(i) for i in range(128)] # 嵌入向量128维

# 挑某个用户（比如5701）， 可视化展示该用户前后记录中item商品向量相似性
tmp = train_click_0_df[train_click_0_df['user_id'] == 5701]
tmp = tmp.merge(out_df, on='item_id', how='left')
nonull_tmp = tmp[~tmp['item_vec0'].isnull()]

sim_list = []
for i in range(0, nonull_tmp.shape[0] - 1):
    emb1 = nonull_tmp.values[i][-128:]
    emb2 = nonull_tmp.values[i + 1][-128:]
    sim_list.append(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * (np.linalg.norm(emb2))))
sim_list.append(0)

plt.figure()
plt.figure(figsize=(10, 6))
fig = sns.lineplot(x=[i for i in range(len(sim_list))], y=sim_list)
for item in fig.get_xticklabels():
    item.set_rotation(90)
plt.tight_layout()
plt.show()

"""
EDA后作者思考：
1.用户前后点击的商品向量 图片向量 文本向量 相似性并不高，所以相似度推荐不一定效果很好。
    （所以虽然说user CF更加倾向于推荐热门商品，item CF倾向推荐长尾，但是CF类算法基于相似度，可能效果不一定很好）
2.前后关联关系，是短期兴趣，可以尝试提取长期兴趣。
3.考虑采用 召回+排序的架构  召回的方法+ 排序的相关方法
"""