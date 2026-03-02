import joblib
import pandas as pd
import numpy as np

def predict_price(features_dict):
    # Load AVM
    avm = joblib.load('../src/api/models/avm.pkl')
    
    # Convert dict to DF, apply engineering (simplified)
    df = pd.DataFrame([features_dict])
    # Apply feature_engineering and geospatial (call functions)
    from feature_engineering import engineer_features
    from geospatial_analysis import add_geospatial_features
    df = engineer_features(df)
    df = add_geospatial_features(df)
    
    pred_log = avm.predict(df)
    return np.expm1(pred_log)[0]

# Example
if __name__ == "__main__":
    sample = {'TotalBsmtSF': 1000, '1stFlrSF': 1000, '2ndFlrSF': 0, 'FullBath': 2, 'YrSold': 2023, 'YearBuilt': 2000}  # Add all features
    print(f"Predicted Price: ${predict_price(sample):,.2f}")
