import pandas as pd
import numpy as np
import os
from datetime import datetime

# 1. 数据加载
print("===== 数据加载 =====")
df = pd.read_excel('../data/raw/order2021.xlsx')
print(f"原始数据形状: {df.shape}")

# 2. 数据预览
print("\n===== 前5行数据 =====")
print(df.head())

# 3. 数据信息
print("\n===== 数据信息 =====")
print(df.info())

# 4. 描述统计
print("\n===== 描述统计 =====")
print(df.describe())

# 5. 数据清洗
print("\n===== 数据清洗 =====")
# 列名去空格
df.columns = df.columns.str.strip()

# 删除重复值
initial_count = len(df)
df.drop_duplicates(inplace=True)
print(f"删除重复值: {initial_count} -> {len(df)} 条记录")

# 缺失值处理
print("\n缺失值统计:")
print(df.isnull().sum())
df.dropna(inplace=True)
print(f"删除缺失值后记录数: {len(df)}")

# 异常值处理 (付款金额为负)
abnormal = df[df['付款金额'] < 0]
print(f"\n发现异常值 {len(abnormal)} 条:")
print(abnormal)
df = df[df['付款金额'] >= 0]
print(f"删除异常值后记录数: {len(df)}")

# 6. 特征工程
print("\n===== 特征工程 =====")
df['订单日期'] = pd.to_datetime(df['付款时间']).dt.date
df['月份'] = pd.to_datetime(df['订单日期']).dt.to_period('M')
df['是否退款'] = df['是否退款'].astype('category')  # 分类变量优化

# 7. 筛选无退款订单
df_no_refund = df[df['是否退款'] == '否'].copy()
df_no_refund['订单日期'] = pd.to_datetime(df_no_refund['订单日期'])

# 8. 数据保存
print("\n===== 数据保存 =====")
output_dir = '../data/processed/'
os.makedirs(output_dir, exist_ok=True)  # 确保目录存在

# 保存全量清洗数据
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
save_path = f"{output_dir}cleaned_data_{timestamp}"

# 多种格式保存（按需选择）
df.to_csv(f"{save_path}.csv", index=False, encoding='utf-8-sig')  # 通用CSV
df.to_pickle(f"{save_path}.pkl")  # 保留完整数据类型
df.to_parquet(f"{save_path}.parquet", engine='pyarrow')  # 高效列式存储

# 保存无退款子集
df_no_refund.to_csv(f"{output_dir}no_refund_orders.csv", index=False)

print(f"""
=== 保存完成 ===
1. 全量清洗数据:
   - CSV: {save_path}.csv
   - Pickle: {save_path}.pkl
   - Parquet: {save_path}.parquet
2. 无退款订单子集:
   - {output_dir}no_refund_orders.csv
""")

# 9. 最终验证
print("\n===== 最终数据验证 =====")
print("全量数据信息:")
print(df.info())
print("\n无退款订单样本:")
print(df_no_refund.head())