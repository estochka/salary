import time
import numpy as np
import pickle

import gc

from catboost import CatBoostRegressor
import pygsheets


# __________SELF___________
import parametrs as p
import predict as pred


if __name__ == "__main__":
    # loading models
    cat_model = CatBoostRegressor()
    cat_model.load_model(p.cat_model_path)

    with open(p.preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)

    with open(p.regress_path, "rb") as f:
        regress = pickle.load(f)

    gc_ = pygsheets.authorize(client_secret='data/token.json')
    # Open spreadsheet and then worksheet
    sh = gc_.open('test_model')
    wks = sh.sheet1

    while True:
        # cells B4:B5 predict marks
        check_predict = np.array(wks.range('B4:B5', returnas='matrix')) != ''
        if any(check_predict):
            wks.update_value('B2', 'Работаю....')
            rows = np.where(check_predict)[0]

            # read data
            raw_data = wks.get_as_df(has_header=False, start='C4', end='E5', empty_value=None)
            raw_data.columns = p.models_features

            # choose data for predict
            raw_data = raw_data.iloc[rows]

            # _____PREDICTED PART______
            # catboost
            cat_pred = np.expm1(pred.cat_predict(raw_data, cat_model))

            # regress
            regress_pred = pred.regress(raw_data, preprocessor, regress)

            # Cat output
            for i, itm in enumerate(cat_pred):
                wks.update_value(f'F{rows[i] + 4}', round(itm))

            # Cat output
            for i, itm in enumerate(regress_pred):
                wks.update_value(f'G{rows[i] + 4}', round(itm))

            # clear predict marks
            for i in rows:
                wks.update_value(f'B{i + 4}', '')

            wks.update_value('B2', '')
            del cat_pred, raw_data, rows, regress_pred

        wks.update_value('D1', time.strftime('%d.%m.%y %H:%M', time.localtime()))
        del check_predict
        gc.collect()
        time.sleep(30.0)
