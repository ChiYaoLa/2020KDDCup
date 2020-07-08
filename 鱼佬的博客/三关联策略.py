# -*- coding: utf-8 -*-
# @Time    : 2020/6/30 10:36
from _collections import defaultdict
from tqdm import tqdm
"""
三关联特征 = 有向性打分*位置打分*时间打分
强关联的发生是有向的、有位置的和有时间的。
有向的:比如我们先买了手机，那下一次买手机壳的关联，和先买手机壳再买手机的关联，这两种很明显，A到B大于B到A，这是有向性；
有位置：我先买了手机，然后买了手机壳，又买了耳机，很明显，手机和手机壳的关联性大于手机与耳机的关联性，这是位置性；
有时间的：那么如果再加上时间这层因素，时间相隔越远的关联性肯定是不高的。
"""
def get_sim_item(df_, user_col, item_col, use_iif=False):

    df = df_.copy()
    user_item_ = df.groupby(user_col)[item_col].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_[user_col], user_item_[item_col]))

    user_time_ = df.groupby(user_col)['time'].agg(list).reset_index() # 引入时间因素
    user_time_dict = dict(zip(user_time_[user_col], user_time_['time']))

    sim_item = {}
    item_cnt = defaultdict(int)  # 商品被点击次数
    for user, items in tqdm(user_item_dict.items()):
        for loc1, item in enumerate(items):
            item_cnt[item] += 1
            sim_item.setdefault(item, {})
            for loc2, relate_item in enumerate(items):
                if item == relate_item:
                    continue
                t1 = user_time_dict[user][loc1] # 点击时间提取
                t2 = user_time_dict[user][loc2]
                sim_item[item].setdefault(relate_item, 0)
                if not use_iif:
                    if loc1-loc2>0:
                        sim_item[item][relate_item] += 1 * 0.7 * (0.8**(loc1-loc2-1)) * (1 - (t1 - t2) * 10000) / math.log(1 + len(items)) # 逆向
                    else:
                        sim_item[item][relate_item] += 1 * 1.0 * (0.8**(loc2-loc1-1)) * (1 - (t2 - t1) * 10000) / math.log(1 + len(items)) # 正向
                else:
                    sim_item[item][relate_item] += 1 / math.log(1 + len(items))

    sim_item_corr = sim_item.copy() # 引入AB的各种被点击次数
    for i, related_items in tqdm(sim_item.items()):
        for j, cij in related_items.items():
            sim_item_corr[i][j] = cij / ((item_cnt[i] * item_cnt[j]) ** 0.2)

    return sim_item_corr, user_item_dict