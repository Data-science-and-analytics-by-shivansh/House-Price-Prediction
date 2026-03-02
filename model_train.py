import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
from sklearn.ensemble import StackingRegressor

def train_models():
    df = pd.read_csv('../data/engineered_train.csv')
    X = df.drop('SalePrice', axis=1)
    y = np.log1p(df['SalePrice'])  # Log transform for stability
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Base models
    models = {
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'RF': RandomForestRegressor(n_estimators=100, random_state=42),
        'GB': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGB': xgb.XGBRegressor(n_estimators=100, random_state=42),
        'LGB': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    }
    
    # Train bases
    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, f'../src/api/models/{name}.pkl')
    
    # Stacking for AVM
    stack = StackingRegressor(estimators=list(models.items()), final_estimator=Ridge())
    stack.fit(X_train, y_train)
    joblib.dump(stack, '../src/api/models/avm.pkl')
    
    # Evaluate
    preds = stack.predict(X_test)
    mape = mean_absolute_percentage_error(np.expm1(y_test), np.expm1(preds))
    r2 = r2_score(y_test, preds)
    print(f"AVM - R²: {r2:.3f}, MAPE: {mape*100:.1f}%")

if __name__ == "__main__":
    train_models()
