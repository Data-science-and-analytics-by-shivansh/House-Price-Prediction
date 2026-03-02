import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_data(file_path):
    return pd.read_csv(file_path)

def engineer_features(df):
    # Basic cleaning
    df.fillna(0, inplace=True)
    
    # 100+ Features: Examples (expand as needed)
    # Numerical transformations
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodelAge'] = df['YrSold'] - df['YearRemodAdd']
    df['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath'] + df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
    
    # Interactions (e.g., quality * size)
    df['OverallQual_SF'] = df['OverallQual'] * df['TotalSF']
    df['GrLivArea_Qual'] = df['GrLivArea'] * df['OverallQual']
    
    # Categorical encoding
    cat_cols = ['Neighborhood', 'BldgType', 'HouseStyle', 'Exterior1st', 'SaleType']  # Add more
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = pd.DataFrame(encoder.fit_transform(df[cat_cols]), index=df.index)
    encoded.columns = encoder.get_feature_names_out(cat_cols)
    df = pd.concat([df.drop(cat_cols, axis=1), encoded], axis=1)
    
    # Scaling numerical features
    num_cols = df.select_dtypes(include=[np.number]).columns.drop('SalePrice')  # Exclude target
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    # Add more features: Polynomials, logs, etc.
    for col in ['LotArea', 'GrLivArea', 'TotalSF']:
        df[f'{col}_log'] = np.log1p(df[col])
        df[f'{col}_sq'] = df[col] ** 2
    
    return df

# Example usage
if __name__ == "__main__":
    train = load_data('../data/train.csv')
    engineered = engineer_features(train)
    engineered.to_csv('../data/engineered_train.csv', index=False)
