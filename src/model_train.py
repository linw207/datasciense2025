import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error, roc_curve
from sklearn.preprocessing import StandardScaler
import numpy as np # 引入numpy用于排序

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"  # 请确保路径正确
my_font = FontProperties(fname=font_path)

def train_and_evaluate(features_path):
    features_df = pd.read_csv(features_path)
    os.makedirs('output', exist_ok=True)

    # --- 1. 分类：是否复购 ---
    print("--- 训练复购预测模型 ---")
    X_repeat_cols_df = features_df.drop(['用户名', 'repeat', 'is_churn'], axis=1, errors='ignore')
    if 'future_amount' in X_repeat_cols_df.columns:
        X_repeat_cols_df = X_repeat_cols_df.drop('future_amount', axis=1, errors='ignore')

    X_repeat_processed = pd.get_dummies(X_repeat_cols_df, dummy_na=False)
    y_repeat = features_df['repeat'].astype(int)

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_repeat_processed, y_repeat, test_size=0.2, random_state=42
    )

    scaler_r = StandardScaler()
    X_train_r_scaled = scaler_r.fit_transform(X_train_r)
    X_test_r_scaled = scaler_r.transform(X_test_r)

    clf_r = LogisticRegression(max_iter=1000, random_state=42)
    clf_r.fit(X_train_r_scaled, y_train_r)
    y_pred_r_proba = clf_r.predict_proba(X_test_r_scaled)[:, 1]
    auc_repeat = roc_auc_score(y_test_r, y_pred_r_proba)
    print(f'复购AUC: {auc_repeat:.4f}')

    # 画ROC曲线 - 复购
    fpr_r, tpr_r, _ = roc_curve(y_test_r, y_pred_r_proba)
    plt.figure()
    plt.plot(fpr_r, tpr_r, label=f'ROC曲线 (AUC = {auc_repeat:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('假阳性率', fontproperties=my_font)
    plt.ylabel('真阳性率', fontproperties=my_font)
    plt.title('复购预测ROC曲线', fontproperties=my_font)
    plt.legend(loc='lower right', prop=my_font)
    plt.savefig('output/repeat_roc.png')
    plt.close()

    # --- 2. 分类：是否流失 ---
    print("\n--- 训练流失预测模型 ---")
    # 为流失模型移除 Recency 和 月份 特征以避免数据泄露
    # 使用 errors='ignore' 确保即使某列不存在（例如在调试不同版本的特征文件时）也不会报错
    cols_to_drop_churn = ['用户名', 'repeat', 'is_churn', 'Recency', '月份']
    if 'future_amount' in features_df.columns: # 检查 'future_amount' 是否存在于原始df
        cols_to_drop_churn.append('future_amount')
    
    X_churn_cols_df = features_df.drop(columns=cols_to_drop_churn, errors='ignore')

    X_churn_processed = pd.get_dummies(X_churn_cols_df, dummy_na=False)
    y_churn = features_df['is_churn'].astype(int)
    
    print("用于流失模型的特征列 (独热编码后):", X_churn_processed.columns.tolist()) # 打印特征列

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_churn_processed, y_churn, test_size=0.2, random_state=42
    )

    scaler_c = StandardScaler()
    X_train_c_scaled = scaler_c.fit_transform(X_train_c)
    X_test_c_scaled = scaler_c.transform(X_test_c)

    clf_c = LogisticRegression(max_iter=1000, random_state=42)
    clf_c.fit(X_train_c_scaled, y_train_c)
    y_pred_c_proba = clf_c.predict_proba(X_test_c_scaled)[:, 1]
    auc_churn = roc_auc_score(y_test_c, y_pred_c_proba)
    print(f'流失AUC: {auc_churn:.4f}')

    # 画ROC曲线 - 流失
    fpr_c, tpr_c, _ = roc_curve(y_test_c, y_pred_c_proba)
    plt.figure()
    plt.plot(fpr_c, tpr_c, label=f'ROC曲线 (AUC = {auc_churn:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('假阳性率', fontproperties=my_font)
    plt.ylabel('真阳性率', fontproperties=my_font)
    plt.title('流失预测ROC曲线', fontproperties=my_font)
    plt.legend(loc='lower right', prop=my_font)
    plt.savefig('output/churn_roc.png')
    plt.close()

    # --- 3. 回归：未来消费金额（如果特征文件中包含 'future_amount' 列） ---
    if 'future_amount' in features_df.columns:
        print("\n--- 训练未来消费金额预测模型 ---")
        cols_to_drop_amount = ['用户名', 'repeat', 'is_churn', 'future_amount']
        X_amount_cols_df = features_df.drop(columns=cols_to_drop_amount, errors='ignore')
        
        X_amount_processed = pd.get_dummies(X_amount_cols_df, dummy_na=False)
        y_amount = features_df['future_amount']
        
        print("用于金额预测模型的特征列 (独热编码后):", X_amount_processed.columns.tolist()) # 打印特征列


        X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
            X_amount_processed, y_amount, test_size=0.2, random_state=42
        )

        scaler_a = StandardScaler()
        X_train_a_scaled = scaler_a.fit_transform(X_train_a)
        X_test_a_scaled = scaler_a.transform(X_test_a)
        
        reg = XGBRegressor(random_state=42, objective='reg:squarederror')
        reg.fit(X_train_a_scaled, y_train_a)
        y_pred_a = reg.predict(X_test_a_scaled)
        rmse = mean_squared_error(y_test_a, y_pred_a, squared=False)
        print(f'未来消费金额RMSE: {rmse:.2f}')
        
        plt.figure()
        plt.scatter(y_test_a, y_pred_a, alpha=0.5)
        plt.xlabel('真实消费金额', fontproperties=my_font)
        plt.ylabel('预测消费金额', fontproperties=my_font)
        plt.title('未来消费金额预测散点图', fontproperties=my_font)
        plt.plot([min(y_test_a.min(), y_pred_a.min()), max(y_test_a.max(), y_pred_a.max())],
                 [min(y_test_a.min(), y_pred_a.min()), max(y_test_a.max(), y_pred_a.max())], 
                 'k--', lw=2)
        plt.savefig('output/amount_scatter.png')
        plt.close()
        
        importances = reg.feature_importances_
        feat_names_a = X_amount_processed.columns
        
        sorted_indices = np.argsort(importances)[::-1]
        num_features_to_plot = min(len(feat_names_a), 20)
        
        plt.figure(figsize=(10, max(6, num_features_to_plot * 0.4)))
        plt.barh(feat_names_a[sorted_indices][:num_features_to_plot], 
                 importances[sorted_indices][:num_features_to_plot])
        plt.xlabel('特征重要性', fontproperties=my_font)
        plt.title('XGBoost消费金额预测特征重要性 (Top N)', fontproperties=my_font)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('output/xgb_feature_importance.png')
        plt.close()

if __name__ == "__main__":
    train_and_evaluate('../data/processed/user_features.csv')