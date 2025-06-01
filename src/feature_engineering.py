import pandas as pd

def build_user_features(input_path, output_path):
    df = pd.read_csv(input_path, parse_dates=['付款时间', '订单日期'])

    # RFM特征
    now = df['订单日期'].max()
    rfm = df.groupby('用户名').agg({
        '订单日期': lambda x: (now - x.max()).days,
        '订单号': 'count',
        '付款金额': 'sum'
    }).rename(columns={'订单日期': 'Recency', '订单号': 'Frequency', '付款金额': 'Monetary'}).reset_index()

    # 平台类型、渠道编号、月份（取最近一次订单）
    user_last = df.sort_values('订单日期').groupby('用户名').tail(1)[['用户名', '平台类型', '渠道编号', '月份']]

    # 下单时段、是否周末
    df['hour'] = pd.to_datetime(df['付款时间']).dt.hour
    df['is_weekend'] = pd.to_datetime(df['付款时间']).dt.weekday >= 5
    user_time = df.groupby('用户名').agg({'hour': 'mean', 'is_weekend': 'mean'}).reset_index()

    # 合并特征
    features = rfm.merge(user_last, on='用户名').merge(user_time, on='用户名')

    # 标签：是否复购（30天内有第二单）、是否流失（60天内无订单）
    df = df.sort_values(['用户名', '订单日期'])
    df['next_order'] = df.groupby('用户名')['订单日期'].shift(-1)
    df['days_to_next'] = (df['next_order'] - df['订单日期']).dt.days
    df['is_repeat'] = df['days_to_next'] <= 30
    repeat_label = df.groupby('用户名')['is_repeat'].max().reset_index().rename(columns={'is_repeat': 'repeat'})
    features = features.merge(repeat_label, on='用户名')

    last_order = df.groupby('用户名')['订单日期'].max().reset_index()
    features = features.merge(last_order, on='用户名')
    features['is_churn'] = (now - features['订单日期']).dt.days > 60
    features.drop('订单日期', axis=1, inplace=True)

    features.to_csv(output_path, index=False)
    print(f"用户特征工程完成，已保存到 {output_path}")

if __name__ == "__main__":
    build_user_features(
        input_path='../data/processed/cleaned_data_20250601_1615.csv',
        output_path='../data/processed/user_features.csv'
    )