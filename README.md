## Предсказания зарплаты.

Табличка для предсказаний (пока отключено, настройка сервера)
https://docs.google.com/spreadsheets/d/1Veltt6IHAp4w5Mj0Kle4MTH8fdvBiJ-ang_thmwWWXs/

Папка DOCKER:
- main.py для удаленного сервера и работы с Google Таблицей.
- csv_predict.py для локальной работы (не попадает в докер), предсказания через *.csv
- train.py для локальной работы. обучение моделей на новых данных

Остальное вспомогательные модули, в том числе содержат собственные функции под PipeLine (estimators.py)


part1_prep.ipynb - постановка задачи, EDA.

part2_colab_cat.ipynb - построение модели с помощью CatBoost (градиентный бустинг)

part3_colab_nn.ipynb - построение модели с помощью обработки текстового параметра (BERT) и нейросети (PyTorch)

part4_tfidf.ipynb - смена BERT на TF-IDF и построение новой модели
