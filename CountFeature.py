from utils import get_ngram
# from FeatureGenerator import FeatureGenerator
from collections import defaultdict
from utils import division
import json
from nltk import sent_tokenize
from utils import unpack_pair_generator
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


def get_article_part_count(part, ngram=1):
    """
    get ngram count of article title or body
    :param part: title
    :param ngram:
    :return:
    """
    return [len(t) for t in get_ngram(ngram, part)]


class CountFeatureGenerator(object):
    """
    Generate Count feature and write into csv file.
    """
    def __init__(self, name='countFeatureGenerator' ):
        # super(CountFeatureGenerator, self).__init__(name)
        # self.data = data
        self.pair_news = []
        self.parts = ["title", "body"]
        self.ngrams = ["uni", "bi", "tri"]
        self.count_features_df = pd.DataFrame()
        # self.unpack_pair_generator()

    def process_and_save(self, data):
        # a list of title and body key value pairs
        self.pair_news = unpack_pair_generator(data)
        count_features = {}
        ngrams = {}

        # generate count, unique count, and ratio of unique count and count (unique count / count)
        # of title, body, and uni to tri gram
        for part in self.parts:
            for n, gram in enumerate(self.ngrams):
                ngrams[part + "_" + gram] = \
                    list(get_ngram(n, list(map(lambda x: x[part], self.pair_news))))
                count_features["count_" + part + "_" + gram] = \
                    get_article_part_count(list(map(lambda x: x[part], self.pair_news)), n)
                count_features["count_unique_" + part + "_" + gram] = \
                    get_article_part_count(list(map(lambda x: set(x[part]), self.pair_news)), n)
                count_features["ratio_of_unique_" + part + "_" + gram] = \
                    list(map(lambda x, y: division(x, y),
                             count_features["count_unique_" + part + "_" + gram],
                             count_features["count_" + part + "_" + gram]))
        # count of ngram title in body,
        # ratio of ngram title in body (count of ngram title in body / count of ngram title)
        for gram in self.ngrams:
            count_features["count_of_title_"+gram+"_in_body"] = \
                list(map(lambda x, y:
                         sum([1. for word in x if word in set(y)]),
                         ngrams["title_"+gram], ngrams["body_"+gram]))
            count_features["ratio_of_title_" + gram + "_in_body"] = \
                list(map(division,
                         count_features["count_of_title_"+gram+"_in_body"],
                         count_features["count_title_"+gram]))

        # get label of each news and count number of sentence in title and body
        with open("data.json", mode="r") as f:
            data = json.load(f)
            label = []
            len_sent_title = []
            len_sent_body = []
            for news in data:
                label.append(news["label"])
                len_sent_title.append(len(sent_tokenize(news["title"])))
                len_sent_body.append(len(sent_tokenize(news["body"])))
            count_features["label"] = label
            count_features["len_sent_title"] = len_sent_title
            count_features["len_sent_body"] = len_sent_body
            # count_features["label"] = [news["label"] for news in data]
            # count_features["len_sent_title"] = [len(sent_tokenize(news["title"])) for news in data]
            # count_features["len_sent_body"] = [len(sent_tokenize(body))for _, body in data]
        self.count_features_df = pd.DataFrame.from_dict(count_features)
        pd.DataFrame.to_csv(self.count_features_df, "count_feature.csv")

        # pd.DataFrame.to_csv("")
        # self.count_features["len_sent_title"] = [len(sent_tokenize(" ".join(news["title"]))) for news in self.pair_news]
        # self.count_features["len_sent_body"] = [len(sent_tokenize(" ".join(news["body"]))) for news in self.pair_news]

        # train_set = {}
        # test_set = {}
        # for feature in self.count_features.keys():
        #     train, test = train_test_split(self.count_features[feature], test_size=0.2)
        #     train_set[feature] = train
        #     test_set[feature] = test
        #
        # with open("train_count_feature.json", mode="w+") as f:
        #     json.dump(train_set, indent=4, fp=f)
        #
        # with open("test_count_feature.json", mode="w+") as f:
        #     json.dump(test_set, indent=4, fp=f)
        # with open("count_feature.json", mode="w+") as f:
        #     json.dump(self.count_features, indent=4, fp=f)


    def read(self):
        """
        read directly from feature file and split train test set and make prediction using 20% test set
        """
        df = pd.read_csv('count_feature.csv')
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




# if __name__ == '__main__':
#     cf = CountFeatureGenerator()
#     cf.read()