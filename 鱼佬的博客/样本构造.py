# -*- coding: utf-8 -*-
# @Time    : 2020/6/30 10:53

import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import math


# fill user to 50 items
def get_predict(df, pred_col, top_fill):
    top_fill = [int(t) for t in top_fill.split(',')]
    scores = [-1 * i for i in range(1, len(top_fill) + 1)]
    ids = list(df['user_id'].unique())
    fill_df = pd.DataFrame(ids * len(top_fill), columns=['user_id'])
    fill_df.sort_values('user_id', inplace=True)
    fill_df['item_id'] = top_fill * len(ids)
    fill_df[pred_col] = scores * len(ids)
    df = df.append(fill_df)
    df.sort_values(pred_col, ascending=False, inplace=True)
    df = df.drop_duplicates(subset=['user_id', 'item_id'], keep='first')
    df['rank'] = df.groupby('user_id')[pred_col].rank(method='first', ascending=False)
    df = df[df['rank'] <= 50]
    df = df.groupby('user_id')['item_id'].apply(lambda x: ','.join([str(i) for i in x])).str.split(',',
                                                                                                   expand=True).reset_index()
    return df


now_phase = 4
train_path = './data/underexpose_train'
test_path = './data/underexpose_test'
recom_item = []

whole_click = pd.DataFrame()
for c in range(now_phase + 1):
    print('phase:', c)
    click_train = pd.read_csv(train_path + '/underexpose_train_click-{}.csv'.format(c), header=None,
                              names=['user_id', 'item_id', 'time'])
    click_test = pd.read_csv(test_path + '/underexpose_test_click-{}.csv'.format(c, c), header=None,
                             names=['user_id', 'item_id', 'time'])

    all_click = click_train.append(click_test)
    whole_click = whole_click.append(all_click)
    whole_click = whole_click.drop_duplicates(subset=['user_id', 'item_id', 'time'], keep='last')
    whole_click = whole_click.sort_values('time')

    item_sim_list, user_item = get_sim_item(whole_click, 'user_id', 'item_id', use_iif=False)

    for i in tqdm(click_test['user_id'].unique()):
        rank_item = recommend(item_sim_list, user_item, i, 500, 500)
        for j in rank_item:
            recom_item.append([i, j[0], j[1]])

        # find most popular items
top50_click = whole_click['item_id'].value_counts().index[:50].values
top50_click = ','.join([str(i) for i in top50_click])

recom_df = pd.DataFrame(recom_item, columns=['user_id', 'item_id', 'sim'])
result = get_predict(recom_df, 'sim', top50_click)
result.to_csv('baseline.csv', index=False, header=None)