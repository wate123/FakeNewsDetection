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
    """
    Generate Sentiment feature and write into csv file.
    """
    def __init__(self):
        # self.data = data
        self.pair_news = []
        # self.sentiment_feature_df = pd.DataFrame()
        self.sid = SentimentIntensityAnalyzer()

    def compute_sentiment(self, sentences):
        """
        Feed in title sentences.
        Compute polarity scores and average them
        :param sentences: title in sentence format
        :return: average of polarity scores
        """
        results = [self.sid.polarity_scores(sentence)for sentence in sentences]
        # print(np.mean(results, axis=0).reshape(-1).shape)
        # print(np.reshape(np.mean(results, axis=0),(1, 4)))
        # result = np.reshape(np.mean(results, axis=0), (1, 4))
        return pd.DataFrame(results).mean()

    def process_and_save(self):
        """
        Generate title and body polarity score for each news
        :return:
        """
        print("Generate Title Sentiment Features")
        self.pair_news = pd.read_json("data.json")

        title_sentiment_feature_df = pd.DataFrame()
        body_sentiment_feature_df = pd.DataFrame()

        title_sentiment_feature_df["title_sent"] = self.pair_news["title"].apply(lambda x: sent_tokenize(x))
        title_sentiment_feature_df = pd.concat(
            [title_sentiment_feature_df,
             title_sentiment_feature_df["title_sent"]
                 .apply(lambda x: self.compute_sentiment(x))], axis=1)
        title_sentiment_feature_df.rename(
            columns={'compound': 'title_compound', 'neg': 'title_neg', 'neu': 'title_neu', 'pos': 'title_pos'},
            inplace=True)
        title_sentiment_feature_df["label"] = self.pair_news["label"].tolist()
        title_sentiment_feature_df.drop("title_sent", axis=1).to_csv("title_sentiment_feature.csv")
        print("Article title Done!")
        print("Save into title_sentiment_feature.csv")
        print()
        print("Generate Title Sentiment Features")
        body_sentiment_feature_df["body_sent"] = self.pair_news["body"].apply(lambda x: sent_tokenize(x))
        body_sentiment_feature_df = pd.concat(
            [body_sentiment_feature_df,
             body_sentiment_feature_df["body_sent"]
                 .apply(lambda x: self.compute_sentiment(x))], axis=1)
        body_sentiment_feature_df.rename(
            columns={'compound': 'body_compound', 'neg': 'body_neg', 'neu': 'body_neu', 'pos': 'body_pos'},
            inplace=True)
        body_sentiment_feature_df["label"] = self.pair_news["label"].tolist()
        body_sentiment_feature_df.drop("body_sent", axis=1).to_csv("body_sentiment_feature.csv")
        print("Article body Done!")
        print("Save into body_sentiment_feature.csv")


        # with open('sentiment_feature.json', mode="w+") as f:
        #     json.dump(self.sentiment_feature, indent=4, fp=f)
    def read(self):
        """
        TODO not sure how the sentiment feature feed into model
        :return:
        """
        df = pd.read_csv('title_sentiment_feature.csv')
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
