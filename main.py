from utils import NewsContent
import json
from gensim.corpora.dictionary import Dictionary
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from gensim.models.word2vec import LineSentence, PathLineSentences
from gensim.utils import save_as_line_sentence
import numpy as np
import multiprocessing
from utils import tsne_similar_word_plot, get_ngram
from Word2VecFeature import Word2VecFeatureGenerator
from CountFeature import CountFeatureGenerator
from SentimentFeature import SentimentFeatureGenerator
from TfidfFeature import TfidfFeature
from SvdFeature import SvdFeature
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn import utils, metrics
from sklearn.preprocessing import scale, normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV as GCV
import time

# call NewsContent class to preprocess/tokenize the news content
data = NewsContent('../FakeNewsNet/code/fakenewsnet_dataset', ['politifact'], ['fake', 'real'])
save_as_line_sentence(data.get_features(), "news_corpus.txt")
data.save_in_sentence_form()

# feature_generator = [CountFeatureGenerator(), SentimentFeatureGenerator()]
feature_generator = [CountFeatureGenerator(), SentimentFeatureGenerator()]
[g.process_and_save() for g in feature_generator]
w2v = Word2VecFeatureGenerator(LineSentence("news_corpus.txt"))
w2v.process_and_save(data.get_features("pair"))

features = [g.read() for g in feature_generator]
features.append(pd.read_csv('svd_feature.csv').drop("label", axis=1))
features.append(pd.read_csv('w2v_feature.csv'))
print(features)
print('finish feature loading')
# df_final = pd.DataFrame(np.hstack(features))
df_final = pd.concat(features, axis=1)
# df_final = reduce(lambda left, right: pd.merge(left, right, on='label'), features)
# no_scale = df_concat.drop("label", axis=1)
X = scale(normalize(np.nan_to_num(df_final.drop("label", axis=1).values)))
print("shape: ", X.shape)
# X = df.drop("label", axis=1).values
y = pd.read_csv("data.csv")["label"]
X, y = utils.shuffle(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# grid_C = [0.5 * i for i in range(1, 21)]
# parameters = {"tol": [5e-4], "C": grid_C, "random_state": [1],
#               "solver": ["newton-cg", "sag", "saga", "lbfgs"],
#               "max_iter": [4000], "multi_class": ["multinomial", "ovr", "auto"]}

grid_R = [0.1 * i for i in range(1, 10)]
grid_N = [10 * i for i in range(1, 21)]
# clf = LogisticRegression(tol=0.0005, C=0.5, max_iter=4000, multi_class='ovr', random_state=1, solver='newton-cg')
# parameters = {'loss': 'deviance', 'max_depth': 80, 'max_features': 'auto', 'min_samples_leaf': 4, 'min_samples_split': 4, 'n_estimators': 1000, 'random_state': 42}
parameters = {'max_depth': [3], 'learning_rate': grid_R, 'n_estimators': grid_N, 'silent': [True],
                   'objective': ['multi:softmax', 'multi:softprob'], 'booster':['gbtree', 'dart'],
                   'subsample':[0.7, 0.8, 0.9, 1.], 'random_state': [0], "num_class":[2]}
# clf = GradientBoostingClassifier()
clf = xgb.XGBClassifier()

# print("start hyperperameter tuning")
print(time.time())
# clf = GCV(LogisticRegression(), parameters, cv=2, n_jobs=40)
# clf = GCV(GradientBoostingClassifier(), parameters, cv=2, n_jobs=40)
# clf = GCV(xgb.XGBClassifier(), parameters, cv=2, n_jobs=40)

clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)
print(time.time())
tpfptnfn = metrics.confusion_matrix(y_test, y_predict)
preRecF1 = metrics.classification_report(y_test, y_predict)
# print(clf.best_params_)
print(preRecF1)


