import pandas as pd
import numpy as np
import pickle

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import make_column_transformer

from catboost import Pool, CatBoostRegressor

# __________SELF___________
import estimators as est
import parametrs as p

RAND = sum(ord(x) for x in 'NEVER SURRENDER')

if __name__ == "__main__":
    # overwrite incl
    # mode
    save_prep = False
    save_cat = False
    save_regressor = False  # save_prep required

    df = pd.read_csv('data/train.csv', index_col=[0])
    df.columns = df.columns.str.replace(r'(.)([A-Z])', r'\1_\2', regex=True).str.lower()

    # no need to split. train.csv is only for train
    if save_prep:

        # create pipeline for categorical features
        categorical_pipeline = Pipeline([
            ('loc_category', est.LocationCategory('location_normalized', 10)),
            ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('encoder', OneHotEncoder(handle_unknown='ignore')),
        ])

        # create pipeline for text
        text_pipeline = Pipeline([('clear', est.TextClear()),
                                  # ('lemma', TextLemma(var.nlp)),
                                  ('tfidf', TfidfVectorizer(stop_words='english', min_df=5)),
                                  ])

        col_transform = make_column_transformer((text_pipeline, p.text_features),
                                                (categorical_pipeline, p.categorical_features),
                                                (StandardScaler(), ['word_num'])
                                                )

        preprocessor = Pipeline([('add_num', est.WordsNumber('full_description')),
                                 ('columns', col_transform)])

        preprocessor.fit(df)

        with open(p.preprocessor_path, 'wb') as f:
            pickle.dump(preprocessor, f)

        if save_regressor:
            features = preprocessor.transform(df)
            regress_model = Ridge(random_state=RAND).fit(features, df[p.target])

            with open(p.regress_path, 'wb') as f:
                pickle.dump(regress_model, f)

    # TIME(!)
    if save_cat:
        catboost_params = {'eval_metric': 'RMSE',
                           'depth': 5,
                           'random_seed': RAND,
                           'learning_rate': 0.12,
                           'iterations': 12000
                           }

        df['full_description'] = est.TextClear().transform(df['full_description'])

        train_pool = Pool(df[p.models_features],
                          np.log1p(df[p.target]),
                          cat_features=p.models_features,
                          text_features=p.text_features
                          )

        cat_model = CatBoostRegressor(**catboost_params)
        cat_model.fit(train_pool)

        cat_model.save_model(p.cat_model_path)
