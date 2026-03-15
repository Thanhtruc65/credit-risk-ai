import pandas as pd
import joblib

def test_extreme_case():
    model = joblib.load('models/loan_model.pkl')
    encoders = joblib.load('models/label_encoders.pkl')
    
    data = {
        'NAME_CONTRACT_TYPE': 'Cash loans',
        'CODE_GENDER': 'M',
        'FLAG_OWN_CAR': 'N',
        'FLAG_OWN_REALTY': 'N',
        'CNT_CHILDREN': 4,
        'AMT_INCOME_TOTAL': 20000,
        'AMT_CREDIT': 900000,
        'AMT_ANNUITY': 70000,
        'NAME_EDUCATION_TYPE': 'Lower secondary',
        'NAME_FAMILY_STATUS': 'Single / not married',
        'NAME_HOUSING_TYPE': 'Rented apartment',
        'DAYS_EMPLOYED': 10,
        'AGE': 19
    }
    
    df = pd.DataFrame([data])
    cols = [
        'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 
        'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 
        'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 
        'DAYS_EMPLOYED', 'AGE'
    ]
    df = df[cols]
    
    print("Pre-encoding:\n", df.iloc[0])
    
    cat_cols = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 
                'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE']
    for col in cat_cols:
        df[col] = encoders[col].transform(df[col].astype(str))
        
    print("\nPost-encoding:\n", df.iloc[0])
    
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0]
    
    print("\nPrediction:", pred)
    print("Probabilities:", proba)

if __name__ == "__main__":
    test_extreme_case()
