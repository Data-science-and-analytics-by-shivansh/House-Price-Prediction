import pandas as pd

def compute_neighborhood_indices(df):
    # Group by neighborhood
    grouped = df.groupby('Neighborhood').agg({
        'OverallQual': 'mean',
        'SalePrice': 'mean',
        'HouseAge': 'mean',
        'TotalBath': 'mean'
    }).reset_index()
    
    # Normalize and score (0-100)
    for col in ['OverallQual', 'SalePrice', 'TotalBath']:
        grouped[f'{col}_norm'] = 100 * (grouped[col] - grouped[col].min()) / (grouped[col].max() - grouped[col].min())
    grouped['HouseAge_norm'] = 100 * (1 - (grouped['HouseAge'] - grouped['HouseAge'].min()) / (grouped['HouseAge'].max() - grouped['HouseAge'].min()))  # Younger better
    
    grouped['QualityIndex'] = grouped[['OverallQual_norm', 'SalePrice_norm', 'HouseAge_norm', 'TotalBath_norm']].mean(axis=1)
    grouped.sort_values('QualityIndex', ascending=False).to_csv('neighborhood_indices.csv', index=False)
    print("Neighborhood indices saved")

if __name__ == "__main__":
    df = pd.read_csv('../data/engineered_train.csv')
    compute_neighborhood_indices(df)
