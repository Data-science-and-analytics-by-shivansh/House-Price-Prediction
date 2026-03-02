from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from feature_engineering import engineer_features
from geospatial_analysis import add_geospatial_features

app = Flask(__name__)
avm = joblib.load('models/avm.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    df = engineer_features(df)
    df = add_geospatial_features(df)
    pred_log = avm.predict(df)
    price = np.expm1(pred_log)[0]
    return jsonify({'predicted_price': price})

if __name__ == '__main__':
    app.run(debug=True)
