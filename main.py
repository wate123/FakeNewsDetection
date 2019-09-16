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
from classfiers import xgboost, logistic_reg, random_forest, ada_boost, dt, knn, svm
import time, random
import datetime

controls = {"GridSearch": False, "DefaultParams": True}
# # call NewsContent class to preprocess/tokenize the news content
dataset = ['politifact']
# dataset = ['gossipcop']
# dataset = ['politifact', 'gossipcop']
data = NewsContent('../fakenewsnet_dataset', dataset, ['fake', 'real'])
# data = NewsContent('../fakenewsnet_dataset', ['politifact', 'gossipcop'], ['fake', 'real'])
# save_as_line_sentence(data.get_features(), "news_corpus.txt")
out_file_path = data.save_in_sentence_form(dataset)

print(controls)
# list_seed = random.sample(range(1, 100), 5)
if controls["GridSearch"]:
    list_seed = [1]
else:
    list_seed = [1] + random.sample(range(1, 100), 5)
print(list_seed)
for seed in list_seed:
    print("Current seed is "+str(seed))
    print()
    np.random.seed(seed)
    random.seed(seed)
    feature_generator = [CountFeatureGenerator(out_file_path), SentimentFeatureGenerator(out_file_path),
                         SvdFeature(out_file_path, seed)]
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
    # features.pop()
    # df_final_nn = pd.concat(features, axis=1)

    # df_final = pd.read_csv("final_features.csv")

    # normalize and scale data
    X = scale(normalize(np.nan_to_num(df_final.values)))
    scaler = MinMaxScaler()

    # make it all positive
    X = scaler.fit_transform(X)
    print("shape of features: ", X.shape)
    # X = df.drop("label", axis=1).values
    y = pd.read_csv(out_file_path)["label"]
    # df_final['label'] = y
    df_final.to_csv("./Features/" + "-".join(dataset) + "/final_features_ml.csv", index=False)
    # df_final_nn.to_csv("final_features_nn.csv", index=False)

    chi2_selector = SelectKBest(chi2, k=20)
    chi2_selector.fit(X, y)
    best_chi2_index = chi2_selector.get_support(indices=True)
    chi2_score = [chi2_selector.scores_[i] for i in list(best_chi2_index)]

    best_chi2_features = df_final.columns[best_chi2_index]

    print("Chi2 Top 20 Scores")
    print(dict(zip(best_chi2_features, chi2_score)))
    # chi2_scores = pd.DataFrame(data=kbestchi2.scores_, columns=df_final.columns[chi2_selector.get_support(indices=True)])

    # print(kbestchi2.pvalues_)
    # chi2_score, chi2_pval = chi2(X, y)
    # print(sorted(chi2_score, reverse=True)[:30])
    # print(sorted(chi2_pval, reverse=True)[:30])
    f_classif_selector = SelectKBest(f_classif, k=20)
    f_classif_selector.fit(X, y)
    best_f_classif_index = f_classif_selector.get_support(indices=True)
    f_classif_score = [f_classif_selector.scores_[i] for i in list(best_f_classif_index)]

    best_f_classif_features = df_final.columns[best_f_classif_index]

    print("ANOVA F-value Top 20 Scores")
    print(dict(zip(best_f_classif_features, f_classif_score)))
    # X = SelectKBest(chi2, k=400).fit_transform(X, y)
    print(X.shape)
    print(X)

    # split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # sme = SMOTEENN(random_state=seed)
    # X_train, y_train = sme.fit_resample(X_train, y_train)

    class_weights = "balanced"
    # class_weights = False
    # class_weights = {str(index): float(value) for index, value in enumerate(compute_class_weight('balanced', np.unique(y_train), y_train))}
    list_classifier = [logistic_reg, random_forest, ada_boost, dt, knn, svm, xgboost]
    # list_classifier = [svm]

    # clf = xgboost(gcv=True)
    # clf = random_forest(gcv=grid_search)
    for classifier in list_classifier:
        clf = classifier(gcv=controls["GridSearch"], default_param=controls["DefaultParams"], class_weight=class_weights, seed=seed)
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
        if controls["GridSearch"]:
            print(clf.best_params_)
            print(clf.scoring)

