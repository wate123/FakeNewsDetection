from utils import NewsContent
from gensim.models.word2vec import LineSentence
from gensim.utils import save_as_line_sentence
import numpy as np
from Word2VecFeature import Word2VecFeatureGenerator
from CountFeature import CountFeatureGenerator
from SentimentFeature import SentimentFeatureGenerator
from SvdFeature import SvdFeature
import pandas as pd, random
from sklearn import metrics
from sklearn.preprocessing import scale, normalize, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2, SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV as GCV
from imblearn.combine import SMOTEENN
import xgboost as xgb
import time

from joblib import Parallel, delayed
np.random.seed(1)
random.seed(1)

# call NewsContent class to preprocess/tokenize the news content
# data = NewsContent('../FakeNewsNet/code/fakenewsnet_dataset', ['politifact'], ['fake', 'real'])
# data = NewsContent('../FakeNewsNet/code/fakenewsnet_dataset', ['politifact', 'gossipcop'], ['fake', 'real'])
# save_as_line_sentence(data.get_features(), "news_corpus.txt")
# data.save_in_sentence_form()

feature_generator = [CountFeatureGenerator(), SentimentFeatureGenerator(), SvdFeature(), Word2VecFeatureGenerator()]

# [g.process_and_save() for g in feature_generator]
# w2v.process_and_save(data.get_features("pair"))


features = [g.read() for g in feature_generator]
#
print(features)
print('finish feature loading')

# store results in data frame
df_final = pd.concat(features, axis=1)
# df_final = pd.read_csv("final_features.csv")
# y = df_final['label']
# df_final = df_final.drop("label", axis=1)

# normalize and scale data
X = scale(normalize(np.nan_to_num(df_final.values)))
scaler = MinMaxScaler()

# make it all positive
X = scaler.fit_transform(X)
print("shape of features: ", X.shape)
# X = df.drop("label", axis=1).values
y = pd.read_csv("data.csv")["label"]
# df_final['label'] = y
# df_final.to_csv("final_features.csv", index=False)

# X, y = utils.shuffle(X, y, random_state=0)
print(pd.DataFrame(chi2(X, y)))
X = SelectKBest(chi2, k=400).fit_transform(X, y)
print(X.shape)
print(X)

# split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# sme = SMOTEENN(random_state=1)
# X_train, y_train = sme.fit_resample(X_train, y_train)
# grid_C = [0.5 * i for i in range(1, 21)]
# parameters = {"tol": [5e-4], "C": grid_C, "random_state": [1],
#               "solver": ["newton-cg", "sag", "saga", "lbfgs"],
#               "max_iter": [4000], "multi_class": ["multinomial", "ovr", "auto"]}

grid_R = [0.1 * i for i in range(1, 10)]
grid_N = [10 * i for i in range(1, 21)]
# clf = LogisticRegression(tol=0.0005, C=0.5, max_iter=4000, multi_class='ovr', random_state=1, solver='saga')
# clf = xgb.XGBClassifier(booster='gbtree', learning_rate=0.2, max_depth=6, n_estimators=50, num_class=2,
#                         objective='multi:softmax', random_state=1, subsample=0.7, n_jobs=40, verbosity=3, scale_pos_weight=0.31)
# parameters = {'loss': 'deviance', 'max_depth': 80, 'max_features': 'auto', 'min_samples_leaf': 4, 'min_samples_split': 4, 'n_estimators': 1000, 'random_state': 42}
parameters = {'max_depth': [3,6], 'learning_rate': grid_R, 'n_estimators': grid_N,
              'objective': ['multi:softmax', 'multi:softprob'], 'booster': ['gbtree', 'dart'],
              'subsample': [0.7, 0.8, 0.9, 1.], 'random_state': [1], "num_class": [2, 4, 6], "verbosity":[3], "scale_pos_weight":[0.31] }
# clf = GradientBoostingClassifier()
# clf = xgb.XGBClassifier()

print("Start hyperperameter tuning")
print("Start "+str(time.time()))
# clf = GCV(LogisticRegression(), parameters, cv=10, n_jobs=40)
# clf = GCV(GradientBoostingClassifier(), parameters, cv=2, n_jobs=40)
clf = GCV(xgb.XGBClassifier(), parameters, cv=5, n_jobs=40)
# eval_set = [(X_test, y_test)]

clf.fit(X_train, y_train, eval_metric="error", verbose=True)
# print(clf.get_booster().get_score(importance_type="gain"))
# fscore = pd.Series(clf.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
# xgb.plot_importance(clf.get_booster())
# print(fscore)
y_predict = clf.predict(X_test)
print("end "+str(time.time()))

# to obtain results of accuracy, precision, recall and F1 score
tpfptnfn = metrics.confusion_matrix(y_test, y_predict)
preRecF1 = metrics.classification_report(y_test, y_predict)
print(tpfptnfn)
print(preRecF1)
print(clf.best_params_)
print(clf.best_score_)
print(clf.scoring)

