# data
models_features = ['full_description', 'location_normalized', 'category']
target = 'salary_normalized'

text_features = 'full_description'
categorical_features = ['location_normalized', 'category']

# disable this version
# nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# prepare
preprocessor_path = 'models/loc_preprocessor.pkl'

# cat
cat_model_path = 'models/cat_model.bin'

# regress
regress_path = 'models/ridge.pkl'
