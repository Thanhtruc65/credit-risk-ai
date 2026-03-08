import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
import joblib
import os
import matplotlib.pyplot as plt

def objective(trial, X_train, y_train, X_val, y_val, ratio):
    """
    Hàm mục tiêu cho Optuna để tối ưu hóa XGBoost.
    """
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'scale_pos_weight': ratio,
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'auc',
        'early_stopping_rounds': 50
    }

    model = XGBClassifier(**param)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    y_prob = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_prob)
    
    return auc


def tune_and_train_model(processed_path, models_dir, n_trials=20):
    print("="*50)
    print("BẮT ĐẦU TỰ ĐỘNG TỐI ƯU HÓA MÔ HÌNH (AUTO-TUNING VỚI OPTUNA)")
    print("="*50)
    
    print(f"Loading processed data from {processed_path}...")
    if not os.path.exists(processed_path):
        print(f"Error: {processed_path} not found. Run preprocess.py first.")
        return

    df = pd.read_csv(processed_path)
    
    # Chuẩn bị dữ liệu
    X = df.drop(['SK_ID_CURR', 'TARGET'], axis=1)
    y = df['TARGET']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    ratio = float(np.sum(y == 0)) / np.sum(y == 1)
    
    print(f"Bắt đầu tìm kiếm tham số tối ưu ({n_trials} trials)...")
    
    # Tạo study Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, ratio), n_trials=n_trials)
    
    print("\n" + "="*50)
    print("TÌM KIẾM HOÀN TẤT!")
    print(f"ROC-AUC tốt nhất đạt được: {study.best_value:.4f}")
    print("Các tham số tốt nhất:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("="*50)

    # Huấn luyện lại mô hình với tham số tốt nhất
    print("\nĐang huấn luyện lại mô hình bằng cấu hình tốt nhất...")
    best_params = study.best_params
    best_params['scale_pos_weight'] = ratio
    best_params['random_state'] = 42
    best_params['use_label_encoder'] = False
    best_params['eval_metric'] = 'auc'
    best_params['early_stopping_rounds'] = 50
    
    best_model = XGBClassifier(**best_params)
    
    best_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Đánh giá mô hình cuối cùng
    y_pred = best_model.predict(X_val)
    y_prob = best_model.predict_proba(X_val)[:, 1]
    
    print("\n--- Hiệu suất Mô Hình Tối Ưu Nhất ---")
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_val, y_prob):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    # Lưu biểu đồ feature importance
    plt.figure(figsize=(10, 8))
    feat_importances = pd.Series(best_model.feature_importances_, index=X.columns)
    feat_importances.nlargest(20).plot(kind='barh')
    plt.title('Top 20 Features Importance (Optuna Optimized Model)')
    if not os.path.exists('static'): os.makedirs('static')
    plt.savefig('static/feature_importance.png')
    print("Feature importance plot updated at static/feature_importance.png")
    
    # Lưu mô hình
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    joblib.dump(best_model, os.path.join(models_dir, 'loan_model.pkl'))
    print(f"\nMô hình tốt nhất đã được lưu đè vào {os.path.join(models_dir, 'loan_model.pkl')}")

if __name__ == "__main__":
    # Cài đặt Optuna nếu chưa có
    try:
        import optuna
    except ImportError:
        print("Đang cài đặt optuna...")
        os.system("pip install optuna")
        import optuna

    train_filepath = os.path.join(os.path.dirname(__file__), 'data', 'train_processed.csv')
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    
    tune_and_train_model(train_filepath, models_dir, n_trials=30)
