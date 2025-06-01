import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"  # 请确保路径正确
my_font = FontProperties(fname=font_path)

def train_and_evaluate(features_path):
    features = pd.read_csv(features_path)
    X = features.drop(['用户名', 'repeat', 'is_churn'], axis=1)
    X = pd.get_dummies(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 分类：是否复购
    y_repeat = features['repeat'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y_repeat, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)[:,1]
    auc_repeat = roc_auc_score(y_test, y_pred)
    print('复购AUC:', auc_repeat)

    # 画ROC曲线
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    os.makedirs('output', exist_ok=True)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC曲线 (AUC = {auc_repeat:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('假阳性率', fontproperties=my_font)
    plt.ylabel('真阳性率', fontproperties=my_font)
    plt.title('复购预测ROC曲线', fontproperties=my_font)
    plt.legend(loc='lower right', prop=my_font)
    plt.savefig('output/repeat_roc.png')
    plt.close()

    # 分类：是否流失
    y_churn = features['is_churn'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y_churn, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)[:,1]
    auc_churn = roc_auc_score(y_test, y_pred)
    print('流失AUC:', auc_churn)

    # 画ROC曲线
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC曲线 (AUC = {auc_churn:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('假阳性率', fontproperties=my_font)
    plt.ylabel('真阳性率', fontproperties=my_font)
    plt.title('流失预测ROC曲线', fontproperties=my_font)
    plt.legend(loc='lower right', prop=my_font)
    plt.savefig('output/churn_roc.png')
    plt.close()

    # 回归：未来消费金额（如有future_amount字段）
    if 'future_amount' in features.columns:
        y_amount = features['future_amount']
        X_train, X_test, y_train, y_test = train_test_split(X, y_amount, test_size=0.2, random_state=42)
        reg = XGBRegressor()
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        print('RMSE:', rmse)
        # 画真实值与预测值散点图
        plt.figure()
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.xlabel('真实消费金额', fontproperties=my_font)
        plt.ylabel('预测消费金额', fontproperties=my_font)
        plt.title('未来消费金额预测散点图', fontproperties=my_font)
        plt.savefig('output/amount_scatter.png')
        plt.close()
        
        # 特征重要性
        importances = reg.feature_importances_
        feat_names = features.drop(['用户名', 'repeat', 'is_churn', 'future_amount'], axis=1).columns
        plt.figure(figsize=(8, 6))
        plt.barh(feat_names, importances)
        plt.title('XGBoost特征重要性', fontproperties=my_font)
        plt.tight_layout()
        plt.savefig('output/xgb_feature_importance.png')
        plt.close()

if __name__ == "__main__":
    train_and_evaluate('../data/processed/user_features.csv')