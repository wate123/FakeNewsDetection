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
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV as GCV
from imblearn.combine import SMOTEENN
import xgboost as xgb
from classfiers import xgboost, logistic_reg, random_forest, ada_boost, dt, knn, svm
import time
import datetime
from joblib import Parallel, delayed
np.random.seed(1)
random.seed(1)

# # call NewsContent class to preprocess/tokenize the news content
# data = NewsContent('../fakenewsnet_dataset', ['politifact'], ['fake', 'real'])
data = NewsContent('../fakenewsnet_dataset', ['politifact', 'gossipcop'], ['fake', 'real'])
# save_as_line_sentence(data.get_features(), "news_corpus.txt")
data.save_in_sentence_form()

feature_generator = [CountFeatureGenerator(), SentimentFeatureGenerator(),
                     SvdFeature(), Word2VecFeatureGenerator(data.get_features("pair"))]
[g.process_and_save() for g in feature_generator]
# w2v.process_and_save()
# df = CountFeatureGenerator().read()
# count = df["count_body_uni"].values
# uni = df["count_body_uni"].sort_values(ascending=False)
# print(uni)

features = [g.read() for g in feature_generator]
print(features)
print('Finish feature loading')

# store results in data frame
df_final = pd.concat(features, axis=1)


# df_final = pd.read_csv("final_features.csv")

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

print(pd.DataFrame(chi2(X, y)))
X = SelectKBest(chi2, k=400).fit_transform(X, y)
print(X.shape)
print(X)

# split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

sme = SMOTEENN(random_state=1)
X_train, y_train = sme.fit_resample(X_train, y_train)

grid_search = True
class_weights = "balanced"
# class_weights = {str(index): float(value) for index, value in enumerate(compute_class_weight('balanced', np.unique(y_train), y_train))}
# list_classifier = [logistic_reg, random_forest, ada_boost, dt, knn, svm, xgboost]
list_classifier = [xgboost]

# clf = xgboost(gcv=True)
# clf = random_forest(gcv=grid_search)
for classifier in list_classifier:
    clf = classifier(gcv=grid_search, class_weight=False)
    clf.fit(X_train, y_train)
    # print(clf.get_booster().get_score(importance_type="gain"))
    # fscore = pd.Series(clf.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
    # xgb.plot_importance(clf.get_booster())
    # print(fscore)
    y_predict = clf.predict(X_test)
    print("end " + str(datetime.datetime.fromtimestamp(time.time())))

    # to obtain results of accuracy, precision, recall and F1 score
    # tpfptnfn = metrics.confusion_matrix(y_test, y_predict)
    preRecF1 = metrics.classification_report(y_test, y_predict)
    # print(tpfptnfn)
    print(preRecF1)
    if grid_search:
        print(clf.best_params_)
        print(clf.scoring)

