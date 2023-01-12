import pandas as pd

# __________SELF___________
import estimators as est


def cat_predict(data: pd.DataFrame, model) -> pd.Series:
    data['full_description'] = est.TextClear().transform(data['full_description'])
    return model.predict(data)


def regress(data: pd.DataFrame, preprocessor, model):
    features = preprocessor.transform(data)
    return model.predict(features)
