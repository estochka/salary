import time
from time import localtime, strftime
import pickle
import os

import numpy as np
import pandas as pd
import spacy

# from loca import LocationCategory
from modl import TextNet

from catboost import CatBoostRegressor

import torch
import transformers as ppb
import pygsheets


def text_transform(data: pd.Series) -> pd.Series:
    # fist clear text part
    # http https www
    data = data.str.replace(r'((https?:\/\/)|w{3}).*?( |$)', ' ', regex=True)
    data = data.str.replace(r'[^A-Za-z\']', ' ', regex=True).str.lower().str.strip()
    data = data.str.replace(r'\W{2,}', ' ', regex=True)

    return data


def lemmatis(data: pd.Series) -> pd.Series:
    # lemma part
    clear_text = data.apply(lambda row: ' '.join([w.lemma_ for w in nlp(row) if not w.is_stop]))
    return clear_text


def bert_transform(data: pd.Series) -> pd.DataFrame:
    b_token = data.apply((lambda x: tokenizer.encode(x,
                                                     add_special_tokens=True,
                                                     truncation=True)))
    max_len = max(b_token.map(len))
    padded = np.array([i + [0] * (max_len - len(i)) for i in b_token.values])
    attention_mask = np.where(padded != 0, 1, 0)
    embeddings = []
    for j in range(1):
        batch = torch.LongTensor(padded)
        attention_mask_batch = torch.LongTensor(attention_mask)
        with torch.no_grad():
            batch_embeddings = model_bert(batch, attention_mask=attention_mask_batch)
        embeddings.append(batch_embeddings[0][:, 0, :])
    bert_features = np.concatenate(embeddings)
    return pd.DataFrame(bert_features)


def cat_predict(raw_data: pd.DataFrame):
    pred = np.round(cat_model.predict(raw_data))
    return pred


if __name__ == "__main__":

    # variables
    cat_features = ['full_description', 'location_normalized', 'category']
    device = torch.device("cpu")

    # for lemma
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    # bert
    model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel,
                                                        ppb.DistilBertTokenizer,
                                                        'distilbert-base-uncased')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model_bert = model_class.from_pretrained(pretrained_weights)

    # Pipelina cat features
    with open("models/loc_preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    # cat model
    cat_model = CatBoostRegressor()
    cat_model.load_model('models/cat_model.bin')

    # nn model
    nn_model = TextNet(800, 530, 1)
    nn_model.load_state_dict(torch.load('models/nn_model_state.pth', map_location=torch.device('cpu')))
    nn_model.eval()

    # minmax
    with open("models/minmax.pkl", "rb") as f:
        minmax = pickle.load(f)

    gc = pygsheets.authorize(client_secret='data/token.json')
    # Open spreadsheet and then worksheet
    sh = gc.open('test_model')
    wks = sh.sheet1

    while True:
        # cells B4:B5 predict marks
        check_predict = np.array(wks.range('B4:B5', returnas='matrix')) != ''
        if any(check_predict):
            wks.update_value('B2', 'Работаю....')
            rows = np.where(check_predict)[0]
            # read data
            raw_data = wks.get_as_df(has_header=False, start='C4', end='E5', empty_value=None)
            raw_data.columns = cat_features
            # choose data for predict
            raw_data = raw_data.iloc[rows]
            raw_data['full_description'] = text_transform(raw_data['full_description'])
            # ____________________CAT_______________
            # cat can predict on raw data
            cat_pred = cat_predict(raw_data)
            for i, itm in enumerate(cat_pred):
                # write cat predict
                wks.update_value(f'F{rows[i] + 4}', itm)
            # _______________NN________________
            # ooohh bert pls dont die T_T
            clear_descr = lemmatis(raw_data['full_description'])
            bert_features = bert_transform(clear_descr)
            category_features = preprocessor.transform(raw_data)
            nn_test = pd.concat([bert_features, pd.DataFrame(category_features)], axis=1)
            nn_pred = nn_model(torch.tensor(nn_test.values, device=device, dtype=torch.float))

            nn_pred = nn_pred.cpu().detach().numpy().tolist()
            nn_pred = minmax.inverse_transform(nn_pred)

            for i, itm in enumerate(nn_pred):
                wks.update_value(f'G{rows[i] + 4}', *itm)

            for i in rows:
                # clear predict marks
                wks.update_value(f'B{i + 4}', '')
            wks.update_value('B2', '')
        wks.update_value('D1', strftime('%d.%m.%y %H:%M', localtime()))

        time.sleep(30.0)
