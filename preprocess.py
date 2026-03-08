import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import joblib

def aggregate_bureau(df_bureau):
    bureau_agg = df_bureau.groupby('SK_ID_CURR').agg({
        'SK_ID_BUREAU': ['count'],
        'DAYS_CREDIT': ['min', 'max', 'mean'],
        'AMT_CREDIT_SUM': ['sum', 'mean'],
        'AMT_CREDIT_SUM_DEBT': ['sum', 'mean']
    })
    bureau_agg.columns = ['_'.join(col).strip() for col in bureau_agg.columns.values]
    bureau_agg.reset_index(inplace=True)
    
    active_count = df_bureau[df_bureau['CREDIT_ACTIVE'] == 'Active'].groupby('SK_ID_CURR').size().reset_index(name='ACTIVE_LOANS_COUNT')
    bureau_agg = bureau_agg.merge(active_count, on='SK_ID_CURR', how='left')
    bureau_agg['ACTIVE_LOANS_COUNT'] = bureau_agg['ACTIVE_LOANS_COUNT'].fillna(0)
    
    return bureau_agg

def aggregate_previous(df_prev):
    prev_agg = df_prev.groupby('SK_ID_CURR').agg({
        'SK_ID_PREV': ['count'],
        'AMT_ANNUITY': ['mean', 'max'],
        'AMT_APPLICATION': ['mean', 'max'],
        'AMT_CREDIT': ['mean', 'max'],
        'AMT_DOWN_PAYMENT': ['mean'],
        'CNT_PAYMENT': ['mean', 'sum']
    })
    prev_agg.columns = ['_'.join(col).strip() for col in prev_agg.columns.values]
    prev_agg.reset_index(inplace=True)
    
    refused_count = df_prev[df_prev['NAME_CONTRACT_STATUS'] == 'Refused'].groupby('SK_ID_CURR').size().reset_index(name='PREV_REFUSED_COUNT')
    approved_count = df_prev[df_prev['NAME_CONTRACT_STATUS'] == 'Approved'].groupby('SK_ID_CURR').size().reset_index(name='PREV_APPROVED_COUNT')
    
    prev_agg = prev_agg.merge(refused_count, on='SK_ID_CURR', how='left')
    prev_agg = prev_agg.merge(approved_count, on='SK_ID_CURR', how='left')
    
    prev_agg['PREV_REFUSED_COUNT'] = prev_agg['PREV_REFUSED_COUNT'].fillna(0)
    prev_agg['PREV_APPROVED_COUNT'] = prev_agg['PREV_APPROVED_COUNT'].fillna(0)
    
    return prev_agg

def aggregate_installments(df_ins):
    df_ins['PAYMENT_PERC'] = df_ins['AMT_PAYMENT'] / df_ins['AMT_INSTALMENT']
    df_ins['PAYMENT_DIFF'] = df_ins['AMT_INSTALMENT'] - df_ins['AMT_PAYMENT']
    df_ins['DPD'] = df_ins['DAYS_ENTRY_PAYMENT'] - df_ins['DAYS_INSTALMENT']
    df_ins['DPD'] = df_ins['DPD'].apply(lambda x: x if x > 0 else 0)
    
    ins_agg = df_ins.groupby('SK_ID_CURR').agg({
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['mean', 'max'],
        'PAYMENT_DIFF': ['mean', 'sum'],
        'AMT_INSTALMENT': ['mean', 'sum'],
        'AMT_PAYMENT': ['mean', 'sum']
    })
    ins_agg.columns = ['_'.join(col).strip() for col in ins_agg.columns.values]
    ins_agg.reset_index(inplace=True)
    
    return ins_agg

def preprocess_home_credit(data_dir, output_dir):
    print("Loading application data...")
    app_train = pd.read_csv(os.path.join(data_dir, 'application_train.csv'))
    
    keep_cols = [
        'SK_ID_CURR', 'TARGET', 'NAME_CONTRACT_TYPE', 'CODE_GENDER', 
        'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 
        'AMT_CREDIT', 'AMT_ANNUITY', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 
        'NAME_HOUSING_TYPE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'OCCUPATION_TYPE',
        'REGION_RATING_CLIENT', 'CNT_FAM_MEMBERS', 'EXT_SOURCE_2', 'EXT_SOURCE_3'
    ]
    df = app_train[keep_cols].copy()
    
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    df['DAYS_BIRTH'] = df['DAYS_BIRTH'].apply(lambda x: abs(x) // 365)
    df.rename(columns={'DAYS_BIRTH': 'AGE'}, inplace=True)
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].apply(lambda x: abs(x) if pd.notna(x) else x)
    
    bureau_file = os.path.join(data_dir, 'bureau.csv')
    if os.path.exists(bureau_file):
        print("Processing Bureau data...")
        df_bureau = pd.read_csv(bureau_file)
        b_agg = aggregate_bureau(df_bureau)
        df = df.merge(b_agg, on='SK_ID_CURR', how='left')
        del df_bureau, b_agg
    
    prev_file = os.path.join(data_dir, 'previous_application.csv')
    if os.path.exists(prev_file):
        print("Processing Previous Application data...")
        df_prev = pd.read_csv(prev_file)
        p_agg = aggregate_previous(df_prev)
        df = df.merge(p_agg, on='SK_ID_CURR', how='left')
        del df_prev, p_agg
        
    ins_file = os.path.join(data_dir, 'installments_payments.csv')
    if os.path.exists(ins_file):
        print("Processing Installments data...")
        df_ins = pd.read_csv(ins_file)
        i_agg = aggregate_installments(df_ins)
        df = df.merge(i_agg, on='SK_ID_CURR', how='left')
        del df_ins, i_agg

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    agg_cols = [c for c in df.columns if 'count' in c.lower() or 'sum' in c.lower() or 'PREV_' in c or 'ACTIVE_' in c]
    df[agg_cols] = df[agg_cols].fillna(0)
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    for col in num_cols:
        if col != 'TARGET':
            q_high = df[col].quantile(0.999)
            if pd.notna(q_high) and q_high > 1e9:
                df[col] = df[col].clip(upper=q_high)
            df[col] = df[col].fillna(df[col].median())
            
    le_dict = {}
    for col in cat_cols:
        df[col] = df[col].fillna('XNA')
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
        
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df = df.fillna(0)
        
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(le_dict, 'models/label_encoders.pkl')
    joblib.dump(df.drop(['SK_ID_CURR', 'TARGET'], axis=1).columns.tolist(), 'models/feature_columns.pkl')

    if not os.path.exists(output_dir): os.makedirs(output_dir)
    output_path = os.path.join(output_dir, 'train_processed.csv')
    df.to_csv(output_path, index=False)
    print(f"Preprocessing completed. Saved {df.shape[1]} features to {output_path}")

if __name__ == "__main__":
    preprocess_home_credit('data', 'data')
