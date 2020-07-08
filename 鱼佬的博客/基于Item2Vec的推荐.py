# -*- coding: utf-8 -*-
# @Time    : 2020/6/30 16:01

from gensim.models.word2vec import *
import pandas as pd
from time import *
from tqdm import *
import numpy as np
import os
import numpy as np
from tqdm import *
import re
pd.set_option('max_columns',200)


train_dir = '../underexpose_train/'
test_dir = '../underexpose_test/underexpose_test_click-0/'


def get_data(train_dir,test_dir):
    """get data"""
    TrainItemFeat = pd.read_csv(train_dir + 'underexpose_item_feat.csv',header=None)
    TrainItemFeat.iloc[:,1] = TrainItemFeat.iloc[:,1].apply(lambda x:float(x[1:]))
    TrainItemFeat.iloc[:,-1] = TrainItemFeat.iloc[:,-1].apply(lambda x:float(x[:-1]))
    feat_col = ['eb'+str(i) for i in TrainItemFeat.columns[1:]]
    feat_col.insert(0,'items')
    TrainItemFeat.columns = feat_col

    TrainUserFeat = pd.read_csv(train_dir + 'underexpose_user_feat.csv',header=None)
    TrainUserFeat.columns = ['userid','age','gender','city']
    TrainClick = pd.read_csv(train_dir + 'underexpose_train_click-0.csv',header=None)
    TrainClick.columns = ['userid','items','time']
    TrainClick['flag'] = 0
    TestQtime = pd.read_csv(test_dir + 'underexpose_test_qtime-0.csv',header=None)
    TestQtime.columns = ['userid','time']
    TestQtime['flag'] = 1
    TestClick = pd.read_csv(test_dir + 'underexpose_test_click-0.csv',header=None)
    TestClick.columns = ['userid','items','time']
    TestClick['flag'] = 2
    click = pd.concat([TrainClick,TestClick,TestQtime],axis=0 ,sort=False)
    return click,TrainItemFeat,TrainClick,TrainUserFeat,TestQtime,TestClick



#序列 反序列化模型
def save_variable(v,filename):
    f=open(filename,'wb')
    pickle.dump(v,f)
    f.close()
    return filename

def load_variavle(filename):
    f=open(filename,'rb')
    r=pickle.load(f)
    f.close()
    return r

def get_top_similar(items,k = 50):
    """计算item的相似度"""
    re_list = list(map(lambda x:[x[0],x[1]],model.most_similar(positive=[items],topn=k)))
    return re_list

def recommendation(uid):
    """每个用户去重的推荐列表"""
    tg_id = data[data['userid']==uid].copy()
    have_read_list = tg_id['items_last'].values[0]
    re_list = list()
    for i in have_read_list:
        re_list.extend(get_top_similar(i))
        re_list = list(set(re_list)-set(have_read_list))
    return re_list

def multi_process(func_name,process_num,deal_list):
    """
    多线程
    """
    from multiprocessing import Pool
    pool = Pool(process_num)
    result_list = pool.map(func_name,deal_list)
    pool.close()
    pool.join()
    return result_list

def train_model(data):
    """训练模型"""
    begin_time = time()
    model = Word2Vec(data['items'].values, size=1000, window=30, min_count=1, workers=40)
    end_time = time()
    run_time = end_time-begin_time
    print ('该循环程序运行时间：',round(run_time,2)) #该循环程序运行时间： 1.4201874732
    return model

def melt_data():
    z = test_df.groupby(['userid'])['items_last_all'].apply(lambda x:np.concatenate(list(x))).reset_index()
    i = pd.concat([pd.Series(row['userid'], row['items_last_all']) for _, row in z.iterrows()]).reset_index()
    i.columns = ['items_new','userid']
    i['items'] = i['items_new'].apply(lambda x:x[0])
    i['weights'] = i['items_new'].apply(lambda x:x[1])
    return i.iloc[:,1:]


click,TrainItemFeat,TrainClick,TrainUserFeat,TestQtime,TestClick = get_data(train_dir,test_dir)

# 用户的点击物品以及对应的时间，此时，我们更加考虑是否推荐一些与用户曾经点击过的物品类似的东西，
# 这样更有可能点击。试想一下，假如我们最近喜欢新出的iphone，是否很大概率喜欢iphone的一些配件，
# 按照这个道理，我们建立模型。
df = click.merge(TrainUserFeat,on = 'userid',how = 'left')
df = df.merge(TrainItemFeat,on='items',how = 'left')
df = df.sort_values(['userid','time']).reset_index(drop = True)

#重新划分训练集和测试集，并按时间排序，反应用户的点击行为。
train = df[df['flag']!=1].copy()
train = train.sort_values(['userid','time']).reset_index(drop = True)
test = df[df['flag']==1].copy()
test = test.sort_values(['userid','time']).reset_index(drop = True)

"""
训练模型，因为Word2vec的输入是string格式，需要提前处理，同时，把
数据格式处理成uid=['itme1',itme2',...itmen']这种格式，其中items_last为用户点击的
最后3个物品，因为跟时间有关系，我们更加会推与用户最近点击的相关物品
"""
tr = train.copy()
tr['items']  = tr['items'].astype(str)
items_all = tr['items'].unique()
tr = tr.groupby('userid')['items'].apply(lambda x:list(x)).reset_index()
tr['items_last'] = tr['items'].apply(lambda x:x[-3:])
model = train_model(tr)

"""
训练完模型后，我们需要计算用户点击过的物品相似度，一般而言，物品相似度是
根据用户点击过的物品序列，计算embedding，从而计算相似度
"""
recommendation_items = dict()
print('获取相似item')
for i in tqdm(items_all):
    recommendation_items[i] = get_top_similar(i)

#计算用户最后点击的2个物品以及他们所对应的物品，每个物品包括物品代号以及相似度['itmes','sim']
tr['items_last_1'] = tr['items_last'].apply(lambda x:recommendation_items[x[-1]])
tr['items_last_2'] = tr['items_last'].apply(lambda x:recommendation_items[x[-2]])
tr['items_last_all'] = tr['items_last_1']+tr['items_last_2']
#根据相似度排序，越是相似的，优先推荐
tr['items_last_all'] = tr['items_last_all'].apply(lambda x:sorted(x,key = lambda x:x[1],reverse=True))




test_df = test[['userid']].merge(tr,on = 'userid',how = 'left')
test_df = melt_data()
test_df['items'] = test_df['items'].astype(float)
test_df = test_df.merge(TrainUserFeat,on = 'userid',how = 'left')
test_df = test_df.merge(TrainItemFeat,on='items',how = 'left')


test_df = test_df.sort_values(['userid','weights'],ascending=False).reset_index()
submit = test_df.groupby(['userid'])['items'].apply(lambda x:list(x)[:50]).reset_index()
sub = pd.DataFrame(list(submit['items'].values))
sub.columns = ['item_id_'+str(i).zfill(2) for i in range(1,51)]