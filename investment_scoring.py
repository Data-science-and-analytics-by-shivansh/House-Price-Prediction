import pandas as pd
import joblib
import numpy as np

def score_opportunities(df):
    avm = joblib.load('../src/api/models/avm.pkl')
    X = df.drop('SalePrice', axis=1)
    preds_log = avm.predict(X)
    preds = np.expm1(preds_log)
    
    df['PredictedPrice'] = preds
    df['UndervaluedScore'] = (df['PredictedPrice'] - df['SalePrice']) / df['PredictedPrice']
    undervalued = df[df['UndervaluedScore'] > 0.15]  # 15%+ below
    undervalued.to_csv('undervalued_properties.csv', index=False)
    print(f"Found {len(undervalued)} undervalued properties")

if __name__ == "__main__":
    df = pd.read_csv('../data/engineered_train.csv')  # Use test set in prod
    score_opportunities(df)
