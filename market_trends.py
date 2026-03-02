import pandas as pd
from prophet import Prophet

def forecast_trends(df):
    # Assume df has 'YrSold', 'MoSold', 'SalePrice'
    df['Date'] = pd.to_datetime(df['YrSold'].astype(str) + '-' + df['MoSold'].astype(str) + '-01')
    monthly = df.groupby('Date')['SalePrice'].mean().reset_index()
    monthly.columns = ['ds', 'y']
    
    model = Prophet(yearly_seasonality=True)
    model.fit(monthly)
    
    future = model.make_future_dataframe(periods=6, freq='M')  # 6 months
    forecast = model.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(6).to_csv('forecast.csv')
    print("6-month forecast saved to forecast.csv")

if __name__ == "__main__":
    df = pd.read_csv('../data/train.csv')
    forecast_trends(df)
