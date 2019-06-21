from nltk import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json
import statistics
from utils import unpack_pair_generator, division
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


class SentimentFeatureGenerator(object):

    def __init__(self):
        # self.data = data
        self.pair_news = []
        selfsentiment_feature_df = pd.DataFrame()
        self.sid = SentimentIntensityAnalyzer()

    def compute_sentiment(self, sentences):
        results = [list(self.sid.polarity_scores(sentence).values()) for sentence in sentences]
        # print(np.mean(results, axis=0).reshape(-1).shape)
        print(np.reshape(np.mean(results, axis=0),(1, 4)))
        result = np.reshape(np.mean(results, axis=0), (1, 4))
        return result

    def process_and_save(self):
        print("Generate Sentiment Features")
        with open('data.json', mode="r") as f:
            self.pair_news = json.load(f)

        col_name = ['compound', 'neg', 'neu', 'pos']
        title_col = ['title_compound', 'title_neg', 'title_neu', 'title_pos']
        body_col = ['body_compound', 'body_neg', 'body_neu', 'body_pos']

        # title_score = np.empty((0,4))
        # body_score = np.empty((0,4))
        for news in self.pair_news:
            pd.concat([self.sentiment_feature_df, self.compute_sentiment(sent_tokenize(news["title"]))], axis=1)
            pd.concat([self.sentiment_feature_df, self.compute_sentiment(sent_tokenize(news["body"]))], axis=1)
            # np.append(title_score, self.compute_sentiment(sent_tokenize(news["title"])), axis=0)
            # np.append(body_score, self.compute_sentiment(sent_tokenize(news["body"])), axis=0)
            # body_score.append(self.compute_sentiment(sent_tokenize(news["body"])))
        # self.sentiment_feature["title_polarity_score"] = \
        #     [self.compute_sentiment(sent_tokenize(news["title"])) for news in self.pair_news]
        # self.sentiment_feature["body_polarity_score"] = \
        #     [self.compute_sentiment(sent_tokenize(news["body"])) for news in self.pair_news]
        # print(np.asarray(title_score))
        # for count, name in enumerate(col_name):
        # ss = np.asarray(title_score)
        # print(pd.DataFrame(ss, columns=title_col))
        # print(pd.DataFrame(np.asarray(title_score), columns=title_col))
        # print(pd.DataFrame([np.asarray(body_score)], columns=body_col))
        self.sentiment_feature_df = pd.DataFrame(np.asarray(title_score))
            # .merge(pd.DataFrame(np.asarray(body_score), columns=body_score), left_index=True, right_index=True, how="outer")

        self.sentiment_feature_df["label"] = [news["label"] for news in self.pair_news]
        sentiment_feature_df = pd.DataFrame.from_dict(self.sentiment_feature)

        sentiment_feature_df.to_csv("sentiment_feature.csv")
        print("Done!")
        print("Save into sentiment_feature.csv")

        # with open('sentiment_feature.json', mode="w+") as f:
        #     json.dump(self.sentiment_feature, indent=4, fp=f)
    def read(self):
        df = pd.read_csv('sentiment_feature.csv')
        X = df.drop("label", axis=1)
        y = df["label"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        clf = LogisticRegression(solver="saga")

        model = clf.fit(X_train, y_train)

        result = clf.predict(X_test)
        score = metrics.accuracy_score(y_test, result)
        precision = metrics.precision_score(y_test, result, ["fake", "real"], pos_label="real")
        recall = metrics.recall_score(y_test, result, ["fake", "real"], pos_label="real")
        f1 = metrics.f1_score(y_test, result, ["fake", "real"], pos_label="real")
        print("accuracy: ", score)
        print("precision: ", precision)
        print("recall: ", recall)
        print("f1: ", f1)
