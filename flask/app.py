import pickle
import numpy as np

import pandas as pd
from flask import Flask, request, render_template
from catboost import CatBoostRegressor

# __________SELF___________
import parametrs as p
import predict as pred

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    description = request.form["description"]
    location = request.form["location"]
    category = request.form["category"]
    # dataframe
    raw_data = pd.DataFrame([[description, location, category]], columns=p.models_features)

    cat_pred = np.expm1(pred.cat_predict(raw_data, cat_model))
    cat_pred = np.round(cat_pred[0], 2)

    regress_pred = pred.regress(raw_data, preprocessor, regress)
    regress_pred = np.round(regress_pred[0], 2)
    return render_template('index.html', cat_predict=cat_pred, regress_predict=regress_pred)


if __name__ == "__main__":
    cat_model = CatBoostRegressor()
    cat_model.load_model(f'../{p.cat_model_path}')

    with open(f'../{p.preprocessor_path}', "rb") as f:
        preprocessor = pickle.load(f)

    with open(f'../{p.regress_path}', "rb") as f:
        regress = pickle.load(f)

    app.run()
