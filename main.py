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
data = NewsContent('../FakeNewsDetection/fakenewsnet_dataset', ['gossipcop'], ['fake', 'real'])
# data = NewsContent('../FakeNewsNet/code/fakenewsnet_dataset', ['politifact', 'gossipcop'], ['fake', 'real'])
save_as_line_sentence(data.get_features(), "news_corpus.txt")
data.save_in_sentence_form()

# generate features to use
feature_generator = [CountFeatureGenerator(), SentimentFeatureGenerator(), SvdFeature()]
w2v = Word2VecFeatureGenerator(LineSentence("news_corpus.txt"))

# Parallel(n_jobs=-1)(delayed(g.process_and_save) for g in feature_generator)

# call functions to obtain features
[g.process_and_save() for g in feature_generator]
w2v.process_and_save(data.get_features("pair"))

# features = Parallel(n_jobs=-1)(delayed(g.read()) for g in feature_generator)

# store results of features
features = [g.read() for g in feature_generator]
features.append(w2v.read())

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
df_final.to_csv("final_features.csv", index=False)


# select strongest features
X = SelectKBest(chi2, k=400).fit_transform(X, y)
print(X.shape)
print(X)

# split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# dataset is unbalanced need to account for bias so use oversampling to deal with that
sme = SMOTEENN(random_state=1)
X_train, y_train = sme.fit_resample(X_train, y_train)

# best model for predicting fake news with tuned parameters
clf_xg = xgb.XGBClassifier(booster='gbtree', learning_rate=0.2, max_depth=6, n_estimators=50, num_class=2, objective='multi:softmax', random_state=1, silent=True, subsample=0.7, n_jobs=40)


print("Start hyperperameter tuning")
print("Start "+str(time.time()))

# use cross validation and train model
clf = GCV(clf_xg, cv=5, n_jobs=40)
clf.fit(X_train, y_train)

# test model
y_predict = clf.predict(X_test)
print("end "+str(time.time()))

# to obtain results of accuracy, precision, recall and F1 score
tpfptnfn = metrics.confusion_matrix(y_test, y_predict)
preRecF1 = metrics.classification_report(y_test, y_predict)
# print(clf.best_params_)
print(preRecF1)


