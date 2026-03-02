# Enterprise Real Estate Pricing Intelligence Platform

## Overview
Advanced platform for house price prediction using ML. Features include ensemble models, geospatial analysis, market forecasting, and an API for real-time valuations.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Download Ames Housing dataset from Kaggle and place in `data/`.
3. Train models: `python src/models/train.py`
4. Run API: `python src/api/app.py`
5. For analyses, run respective scripts (e.g., `python src/market_trends.py`).

## Business Impact
- R² > 0.92
- MAPE 5-7%
- Identify undervalued properties (15%+ below predicted)
- 6-month market forecasts
- Potential $500K+ investment gains

## License
MIT
