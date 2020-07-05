# -*- coding: utf-8 -*-
# @Time    : 2020/6/30 10:43
"""
交互行为特征：
距离下次点击越近的行为，相关性越高，还可以根据位置远近考虑重要性，添加位置权重因子。
当然还可以添加时间权重因子。
"""

def recommend(sim_item_corr, user_item_dict, user_id, top_k, item_num):
    rank = {}
    interacted_items = user_item_dict[user_id]
    interacted_items = interacted_items[::-1]
    for loc, i in enumerate(interacted_items):
        for j, wij in sorted(sim_item_corr[i].items(), reverse=True)[0:top_k]:
            if j not in interacted_items:
                rank.setdefault(j, 0)
                rank[j] += wij * (0.7 ** loc)

    return sorted(rank.items(), key=lambda d: d[1], reverse=True)[:item_num]