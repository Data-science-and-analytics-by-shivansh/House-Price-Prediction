import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def comparable_sales(target_id, df, top_n=5):
    features = df.drop(['SalePrice', 'Id'], axis=1)
    target = features.loc[df['Id'] == target_id]
    sim = cosine_similarity(target, features)[0]
    df['Similarity'] = sim
    comps = df.sort_values('Similarity', ascending=False).head(top_n + 1)[1:]  # Exclude self
    avg_price = comps['SalePrice'].mean()
    print(f"CMA for ID {target_id}: Avg comparable price ${avg_price:,.2f}")
    return comps

if __name__ == "__main__":
    df = pd.read_csv('../data/engineered_train.csv')
    comparable_sales(1, df)
