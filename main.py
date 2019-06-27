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
import pandas as pd, random
from sklearn import utils, metrics
from sklearn.preprocessing import scale, normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import chi2, SelectKBest, f_classif

from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV as GCV
import time

np.random.seed(1)
random.seed(1)

# call NewsContent class to preprocess/tokenize the news content
data = NewsContent('../FakeNewsNet/code/fakenewsnet_dataset', ['politifact'], ['fake', 'real'])
save_as_line_sentence(data.get_features(), "news_corpus.txt")
data.save_in_sentence_form()

# feature_generator = [CountFeatureGenerator(), SentimentFeatureGenerator()]
feature_generator = [CountFeatureGenerator(), SentimentFeatureGenerator()]
# [g.process_and_save() for g in feature_generator]
# w2v = Word2VecFeatureGenerator(LineSentence("news_corpus.txt"))
# w2v.process_and_save(data.get_features("pair"))

features = [g.read() for g in feature_generator]
features.append(pd.read_csv('svd_feature.csv').drop("label", axis=1))
features.append(pd.read_csv('w2v_feature.csv'))
print(features)
print('finish feature loading')
# df_final = pd.DataFrame(np.hstack(features))
df_final = pd.concat(features, axis=1)

# print(df_final.isnull().sum())
# df_final = reduce(lambda left, right: pd.merge(left, right, on='label'), features)
# no_scale = df_concat.drop("label", axis=1)
X = scale(normalize(np.nan_to_num(df_final.drop("label", axis=1).values)))
print("shape: ", X.shape)
# X = df.drop("label", axis=1).values
y = pd.read_csv("data.csv")["label"]
df_final['label'] = y
df_final.to_csv("final_features.csv")
# X, y = utils.shuffle(X, y, random_state=0)
X = SelectKBest(f_classif, k=500).fit_transform(X, y)
print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

sme = SMOTEENN(random_state=1)
X_train, y_train = sme.fit_resample(X_train, y_train)
# grid_C = [0.5 * i for i in range(1, 21)]
# parameters = {"tol": [5e-4], "C": grid_C, "random_state": [1],
#               "solver": ["newton-cg", "sag", "saga", "lbfgs"],
#               "max_iter": [4000], "multi_class": ["multinomial", "ovr", "auto"]}

grid_R = [0.1 * i for i in range(1, 10)]
grid_N = [10 * i for i in range(1, 21)]
# clf = LogisticRegression(tol=0.0005, C=0.5, max_iter=4000, multi_class='ovr', random_state=1, solver='saga', class_weight="balanced")
clf = xgb.XGBClassifier(booster='gbtree', learning_rate=0.9, max_depth=6, n_estimators=70, num_class=2, objective='multi:softmax', random_state=1, silent=True, subsample=0.7)
# parameters = {'loss': 'deviance', 'max_depth': 80, 'max_features': 'auto', 'min_samples_leaf': 4, 'min_samples_split': 4, 'n_estimators': 1000, 'random_state': 42}
# parameters = {'max_depth': [3,6], 'learning_rate': grid_R, 'n_estimators': grid_N, 'silent': [True],
#                    'objective': ['multi:softmax', 'multi:softprob'], 'booster':['gbtree', 'dart'],
#                    'subsample':[0.7, 0.8, 0.9, 1.], 'random_state': [1], "num_class":[2, 4, 6]}
# clf = GradientBoostingClassifier()
# clf = xgb.XGBClassifier()

print("start hyperperameter tuning")
print("start "+str(time.time()))
# clf = GCV(LogisticRegression(), parameters, cv=10, n_jobs=40)
# clf = GCV(GradientBoostingClassifier(), parameters, cv=2, n_jobs=40)
# clf = GCV(xgb.XGBClassifier(), parameters, cv=10, n_jobs=40)

clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)
print("end "+str(time.time()))
tpfptnfn = metrics.confusion_matrix(y_test, y_predict)
preRecF1 = metrics.classification_report(y_test, y_predict)
# print(clf.best_params_)
print(preRecF1)


