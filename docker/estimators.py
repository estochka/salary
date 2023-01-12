from sklearn.base import BaseEstimator, TransformerMixin


class LocationCategory(BaseEstimator, TransformerMixin):

    def __init__(self, name_of_column, number_of_category):
        self.location_dict = None
        self.name_of_column = name_of_column
        self.number_of_category = number_of_category

    def fit(self, features, y=None):
        self.location_dict = self.create_dict(features[self.name_of_column])
        return self

    def transform(self, features):
        feature = features.copy()
        feature[self.name_of_column] = feature[self.name_of_column].apply(lambda x: self.location_dict.get(x, -1))
        return feature

    def create_dict(self, feature):

        # create groups by value counts
        groups = feature.value_counts()
        # split by ~equals sum per group
        group_sum = round(groups.sum() / self.number_of_category)

        current_sum = 0
        group_num = 0
        group_dict = {}

        for index, itm in zip(groups.index, groups.to_numpy()):
            if (current_sum + itm) > group_sum:
                if abs(current_sum + itm - group_sum) < abs(current_sum - group_sum):
                    # include current itm in group
                    current_sum = 0
                    rem_sum = groups.loc[index:].sum() - itm
                    group_dict[index] = group_num
                    group_num += 1
                    if group_num == (self.number_of_category - 1):
                        break
                else:
                    # exclude current itm
                    group_num += 1
                    rem_sum = groups.loc[index:].sum()
                    if group_num == (self.number_of_category - 1):
                        break
                    current_sum = itm
                    group_dict[index] = group_num

                # refresh group sum
                group_sum = round(rem_sum / (self.number_of_category - group_num))

            else:
                current_sum += itm
                group_dict[index] = group_num
        return group_dict


# pd.Series
class TextClear(BaseEstimator, TransformerMixin):

    def fit(self, feature, y=None):
        return self

    def transform(self, feature, y=None):
        feature = feature.str.replace(r'((https?:\/\/)|w{3}).*?( |$)', ' ', regex=True)
        feature = feature.str.replace(r'[^A-Za-z\']', ' ', regex=True).str.lower().str.strip()
        feature = feature.str.replace(r'\W{2,}', ' ', regex=True)
        return feature


# pd.Series
class TextLemma(BaseEstimator, TransformerMixin):
    # spacy
    def __init__(self, nlp):
        self.nlp = nlp

    def fit(self, feature, y=None):
        return self

    def transform(self, feature, y=None):
        return feature.apply(lambda row: ' '.join([w.lemma_ for w in self.nlp(row) if not w.is_stop]))


class WordsNumber(BaseEstimator, TransformerMixin):

    def __init__(self, name_of_column):
        self.name_of_column = name_of_column

    def fit(self, features, y=None):
        return self

    def transform(self, features, y=None):
        data = features.copy()
        data['word_num'] = data[self.name_of_column].str.count(' ')
        return data
