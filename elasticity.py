import pandas as pd
import joblib
from sklearn.inspection import partial_dependence
import matplotlib.pyplot as plt

def compute_elasticity():
    df = pd.read_csv('../data/engineered_train.csv')
    X = df.drop('SalePrice', axis=1)
    model = joblib.load('../src/api/models/Ridge.pkl')  # Use linear for elasticity
    
    features = ['GrLivArea', 'OverallQual', 'TotalSF']  # Examples
    for feat in features:
        pdp, axes = partial_dependence(model, X, [feat])
        elasticity = (pdp[0][-1] - pdp[0][0]) / (axes[0][-1] - axes[0][0]) * (X[feat].mean() / df['SalePrice'].mean())
        print(f"Elasticity for {feat}: {elasticity:.3f}")
        
        plt.plot(axes[0], pdp[0])
        plt.title(f"Partial Dependence: {feat}")
        plt.savefig(f'elasticity_{feat}.png')

if __name__ == "__main__":
    compute_elasticity()
