import datetime
import os
import random
import time

import numpy as np
import pandas as pd
from joblib import load
from sklearn import metrics
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale, normalize, MinMaxScaler

from CountFeature import CountFeatureGenerator
from NmfFeature import NmfFeature
from SentimentFeature import SentimentFeatureGenerator
from utils import NewsContent

logs = {}


controls = {"GridSearch": False, "DefaultParams": False}
# # call NewsContent class to preprocess/tokenize the news content
# dataset = ['politifact']
# dataset = ['gossipcop']
dataset = ['politifact', 'gossipcop']
data = NewsContent('../fakenewsnet_dataset', dataset, ['fake', 'real'])
out_file_path = data.save_in_sentence_form(dataset)
score_path = "./Results/"+"_".join(dataset)+str(datetime.datetime.now().strftime("%H:%M:%S-%m-%d-%Y"))+"/Eval"
try:
    os.makedirs(score_path)
except OSError:
    raise
logs["Setting"] = controls
logs["DataSet Name"] = dataset
logs["Raw Data Path"] = out_file_path

print(controls)
# list_seed = random.sample(range(1, 100), 5)
list_seed = [1]

print(list_seed)
logs["Random Seeds"] = list_seed
for index, seed in enumerate(list_seed):
    logs["Run "+str(index)] = ""
    print("Current seed is "+str(seed))
    print()
    logs["Current Seed"] = seed
    np.random.seed(seed)
    random.seed(seed)
    feature_generator = [CountFeatureGenerator(out_file_path), SentimentFeatureGenerator(out_file_path),
                         NmfFeature(out_file_path, seed)]
    # for g in feature_generator:
    #     save_path = g.process_and_save()
    #     logs.update(save_path)
    features = [g.read() for g in feature_generator]
    print(features)
    print('Finish feature loading')

    # store results in data frame
    df_final = pd.concat(features, axis=1)
    logs["Number of Articles"] = len(df_final.index)
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
    logs['Final Combined Features'] = "./Features/" + "-".join(dataset) + "/final_features_ml.csv"
    # df_final_nn.to_csv("final_features_nn.csv", index=False)

    chi2_selector = SelectKBest(chi2, k=20)
    chi2_selector.fit(X, y)
    best_chi2_index = chi2_selector.get_support(indices=True)
    chi2_score = [chi2_selector.scores_[i] for i in list(best_chi2_index)]

    best_chi2_features = df_final.columns[best_chi2_index]

    print("Chi2 Top 20 Scores")

    top20_chi2 = pd.DataFrame(sorted(zip(best_chi2_features, chi2_score), key=lambda x: round(x[1],2), reverse=True))
    top20_chi2.to_csv(score_path+"/chi2.csv", header=False, index=False)
    logs["Top 20 Chi2 Test"] = top20_chi2
    print(top20_chi2)

    # X = SelectKBest(chi2, k=400).fit_transform(X, y)
    print(X.shape)
    print(X)

    # split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # sme = SMOTEENN(random_state=seed)
    # X_train, y_train = sme.fit_resample(X_train, y_train)

    class_weights = "balanced"
    # class_weights = False
    list_classifier = []
    logs["Class Weights"] = class_weights
    # clf = xgboost(gcv=True)
    # clf = random_forest(gcv=grid_search)
    scores = {}
    for root, dirs, files in os.walk("./model", topdown=True):
        for f in files:
            if f.endswith(".joblib"):
                list_classifier.append(f)
    for i, classifier in enumerate(list_classifier):
        start_time = time.time()
        clf_name = classifier.split(".")[0]
        clf = load(classifier)

        # clf, clf_name, GCV_param = classifier(gcv=controls["GridSearch"], default_param=controls["DefaultParams"],
        #                                       dataset="-".join(dataset), class_weight=class_weights, seed=seed)
        X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)
        logs["Classifier "+str(i)] = clf_name
        # logs["Grid Search Parameter "+str(i)] = GCV_param
        clf.fit(X_train, y_train)
        # print(clf.get_booster().get_score(importance_type="gain"))
        # fscore = pd.Series(clf.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
        # xgb.plot_importance(clf.get_booster())
        # print(fscore)
        y_predict = clf.predict(X_test)
        end = time.time()
        duration = end - start_time
        logs["Running Duration " +str(i)] = "{} hour {} min {} sec".format(duration // 3600, (duration % 3600) // 60, int(duration % 60))

        print("end " + str(datetime.datetime.fromtimestamp(time.time())))

        # to obtain results of accuracy, precision, recall and F1 score
        # tpfptnfn = metrics.confusion_matrix(y_test, y_predict)
        print(metrics.classification_report(y_test, y_predict))
        preRecF1 = metrics.classification_report(y_test, y_predict, output_dict=True)
        score = {}
        score["Accuracy"] = round(preRecF1["accuracy"], 3)
        score["Precision"] = round(preRecF1["weighted avg"]["precision"],3)
        score["Recall"] = round(preRecF1["weighted avg"]["recall"],3)
        score["F1"] = round(preRecF1["weighted avg"]["f1-score"],3)
        scores[clf_name] = score
        # print(tpfptnfn)
        # scores.append(score)
        print(preRecF1)
        if controls["GridSearch"]:
            print(clf.best_params_)
            logs["Best GCV Params " +str(i)] = clf.best_params_
            print(clf.scoring)
            logs["GCV Prediction "+str(i)] = clf.scoring
    pd.DataFrame().from_dict(scores).transpose().to_csv(score_path+"/Scores.csv")
    # pd.DataFrame(data=scores, columns=["Accuracy", "Precision", "Recall", "F1"]).to_csv(score_path+"/Scores.csv", index=False,)

with open(score_path+"/Experiment_Detail.txt", mode="a") as f:
    for key, val in logs.items():
        f.write(key+": "+str(val)+"\n")

