import pickle
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

with open('data/location_dict.pkl', 'rb') as f:
    location_dict = pickle.load(f)


class LocationCategory(BaseEstimator, TransformerMixin):

    def __init__(self, location_dict):
        # load dict

        self.location_dict = location_dict

    def fit(self, X, y=None):
        # no need to fit, only transform
        return self

    def transform(self, X):
        return pd.DataFrame(X.apply(lambda x: location_dict.get(x, 'few')))
