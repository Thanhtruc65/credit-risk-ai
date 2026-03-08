import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib
import os
import matplotlib.pyplot as plt

def train_xgboost_model(processed_path, models_dir):
    print(f"Loading processed data from {processed_path}...")
    if not os.path.exists(processed_path):
        print(f"Error: {processed_path} not found. Run preprocess.py first.")
        return

    df = pd.read_csv(processed_path)
    
    X = df.drop(['SK_ID_CURR', 'TARGET'], axis=1)
    y = df['TARGET']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    ratio = float(np.sum(y == 0)) / np.sum(y == 1)
    print(f"Negative to Positive Ratio: {ratio:.2f}")

    print("Training XGBoost model...")
    model = XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=ratio,
        random_state=42,
        use_label_encoder=False,
        eval_metric='auc',
        early_stopping_rounds=50
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    
    print("\n--- XGBoost Model Performance ---")
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_val, y_prob):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    plt.figure(figsize=(10, 8))
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(20).plot(kind='barh')
    plt.title('Top 20 Features Importance')
    if not os.path.exists('static'): os.makedirs('static')
    plt.savefig('static/feature_importance.png')
    print("Feature importance plot saved to static/feature_importance.png")
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    joblib.dump(model, os.path.join(models_dir, 'loan_model.pkl'))
    print(f"\nModel saved to {os.path.join(models_dir, 'loan_model.pkl')}")

if __name__ == "__main__":
    train_xgboost_model('data/train_processed.csv', 'models')
