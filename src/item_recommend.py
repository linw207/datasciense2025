import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import matplotlib.pyplot as plt
import os


def run_item_recommend(input_path, target_user=None):
    df = pd.read_csv(input_path)
    user_item = df.pivot_table(index='用户名', columns='商品编号', values='订单号', aggfunc='count', fill_value=0)

    # 只取前1000个用户进行测试
    user_item = user_item.iloc[:1000, :]

    # FP-Growth频繁项集
    basket = user_item.applymap(lambda x: 1 if x > 0 else 0).astype(bool)
    print("basket 统计：")
    print(basket.sum().sort_values(ascending=False).head())
    print("用户数：", basket.shape[0])
    print("商品数：", basket.shape[1])

    frequent_itemsets = fpgrowth(basket, min_support=0.001, use_colnames=True)
    if frequent_itemsets.empty:
        print("没有挖掘到任何频繁项集，请尝试降低min_support或检查数据！")
    else:
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
        print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

    if not frequent_itemsets.empty:
        # 画支持度Top10柱状图
        top_items = frequent_itemsets.sort_values('support', ascending=False).head(10)
        os.makedirs('output', exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.barh(
            [str(i) for i in top_items['itemsets']],
            top_items['support']
        )
        plt.xlabel('支持度')
        plt.title('频繁项集支持度Top10')
        plt.tight_layout()
        plt.savefig('output/frequent_itemsets_top10.png')
        plt.close()
        
    # 协同过滤推荐（只对单个用户计算）
    def recommend(user, topn=5):
        user_vec = user_item.loc[user].values.reshape(1, -1)
        all_vecs = user_item.values
        sims = cosine_similarity(user_vec, all_vecs)[0]
        sim_users_idx = np.argsort(sims)[::-1][1:6]  # 排除自己，取前5
        sim_users = user_item.index[sim_users_idx]
        user_bought = set(user_item.loc[user][user_item.loc[user]>0].index)
        rec_goods = set()
        for u in sim_users:
            rec_goods |= set(user_item.loc[u][user_item.loc[u]>0].index)
        rec_goods -= user_bought
        return list(rec_goods)[:topn]

    if target_user is None:
        target_user = user_item.index[0]  # 默认取第一个用户
    print(f'{target_user} 推荐商品:', recommend(target_user))
    
    rec_list = recommend(target_user)
    print(f'{target_user} 推荐商品:', rec_list)
    # 可视化推荐商品
    plt.figure(figsize=(8, 4))
    plt.bar(rec_list, range(len(rec_list)))
    plt.xlabel('商品编号')
    plt.ylabel('推荐顺序')
    plt.title(f'{target_user} 协同过滤推荐商品')
    plt.tight_layout()
    plt.savefig(f'output/{target_user}_recommend.png')
    plt.close()

if __name__ == "__main__":
    run_item_recommend('../data/processed/cleaned_data_20250601_1615.csv')