import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from loca import LocationCategory

df = pd.read_csv('for_pipe.csv', index_col=[0]).reset_index(drop=True)
with open('data/location_dict.pkl', 'rb') as f:
    location_dict = pickle.load(f)

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False)),
])

location_transform = ColumnTransformer([('loc', LocationCategory(location_dict), 'location_normalized')],
                                       remainder='passthrough')

final_pipe = Pipeline([
    ('loc', location_transform),
    ('prep', cat_pipe)

])

tt = final_pipe.fit(df)
with open("loc_preprocessor.pkl", "wb") as f:
    pickle.dump(tt, f)
