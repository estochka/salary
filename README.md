## Предсказания зарплаты.

Цель: построить модель для удаленного сервера, которая может предсказывать зарплату по резюме.

`pandas` `numpy` `spacy` `pickle` `torch` `sklearn` `catboost` `transformers` `pandas_profiling` 
_________________________
Табличка для предсказаний 
https://docs.google.com/spreadsheets/d/1Veltt6IHAp4w5Mj0Kle4MTH8fdvBiJ-ang_thmwWWXs/

Датасет, который модель "не знает" - https://disk.yandex.ru/d/cKqqDbqp0DUOcw.

- main.py для удаленного сервера и работы с Google Таблицей.
- csv_predict.py для локальной работы (не попадает в докер), предсказания через *.csv
- train.py для локальной работы. обучение моделей на новых данных
- estimators.py - несколько классов на основе BaseEstimator


_________________________________________________________________________________________________________

Папка notebooks
- part1_prep.ipynb - постановка задачи, EDA.
- part2_colab_cat.ipynb - построение модели с помощью CatBoost (градиентный бустинг)
- part3_colab_nn.ipynb - построение модели с помощью обработки текстового параметра (BERT) и нейросети (PyTorch)
- part4_tfidf.ipynb - смена BERT на TF-IDF и построение новой модели
- part5_caterogy.ipynb - решение задачи регрессии через классификацию. плюс более наглядный вывод нейросети из 2 части.
