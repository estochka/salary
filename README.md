## Предсказания зарплаты.

Табличка для предсказаний (пока отключено, настройка сервера)
https://docs.google.com/spreadsheets/d/1Veltt6IHAp4w5Mj0Kle4MTH8fdvBiJ-ang_thmwWWXs/

Датасет, который модель "не знает" - https://disk.yandex.ru/d/cKqqDbqp0DUOcw.

- main.py для удаленного сервера и работы с Google Таблицей.
- csv_predict.py для локальной работы (не попадает в докер), предсказания через *.csv
- train.py для локальной работы. обучение моделей на новых данных
- estimators.py - несколько классов на основе BaseEstimator


Остальное вспомогательные модули, в том числе содержат собственные функции под PipeLine (estimators.py)

_________________________________________________________________________________________________________

Папка notebooks
- part1_prep.ipynb - постановка задачи, EDA.
- part2_colab_cat.ipynb - построение модели с помощью CatBoost (градиентный бустинг)
- part3_colab_nn.ipynb - построение модели с помощью обработки текстового параметра (BERT) и нейросети (PyTorch)
- part4_tfidf.ipynb - смена BERT на TF-IDF и построение новой модели
