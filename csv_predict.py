import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor

# __________SELF___________
import parametrs as p
import predict as pred

import gc
gc.enable()

def metrics(real, pred) -> None:
    print(f'RMSE = {mean_squared_error(real, pred) ** .5}')
    print(f'R2 = {r2_score(real, pred)}')
    print(f'MAE = {mean_absolute_error(real, pred)}')
    return None


if __name__ == "__main__":

    # loading models
    cat_model = CatBoostRegressor()
    cat_model.load_model(p.cat_model_path)

    with open(p.preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)

    with open(p.regress_path, "rb") as f:
        regress = pickle.load(f)

    df = pd.read_csv('data/for_predict.csv')
    df.columns = df.columns.str.replace(r'(.)([A-Z])', r'\1_\2', regex=True).str.lower()

    try:
        y_true = df[p.target]
    except:
        print('Without metrics')

    df = df[p.models_features]

    cat_pred = pred.cat_predict(df, cat_model)

    regress_pred = pred.regress(df, preprocessor, regress)

    try:
        print('\nCatBoost')
        metrics(y_true, np.expm1(cat_pred))
        print('\nRegressor')
        metrics(y_true, regress_pred)
        print('\nBoth')
        metrics(y_true, (np.expm1(cat_pred)*1 + regress_pred*0.5)/1.5)
        df['real'] = y_true
    except:
        pass

    df['predict_cat'] = np.round(np.expm1(cat_pred))
    df['predict_regress'] = np.round(regress_pred)
    df['both'] = (df['predict_cat'] + df['predict_regress'] * 0.5) / 1.5

    df.to_csv('data/predicted.csv')


