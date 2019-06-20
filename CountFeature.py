from utils import get_ngram
# from FeatureGenerator import FeatureGenerator
from collections import defaultdict
from utils import division
import json
from nltk import sent_tokenize
from utils import unpack_pair_generator
from FeatureGenerator import FeatureGenerator
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class CountFeatureGenerator(object):
    # def __init__(self):
    def __init__(self, name='countFeatureGenerator' ):
        # super(CountFeatureGenerator, self).__init__(name)
        # self.data = data
        self.pair_news = []
        self.parts = ["title", "body"]
        self.ngrams = ["uni", "bi", "tri"]
        self.count_features_df = pd.DataFrame()
        # self.unpack_pair_generator()

    def get_article_part_count(self, part, ngram=1):
        # return [len() for count in get_ngram(ngram, self.data)]
        # pair_dict = dict((title, body) for title, body in self.data)
        # print(pair_dict)
        return [len(t) for t in get_ngram(ngram, part)]

    # def unpack_pair_generator(self):
    #     for count, (title, body) in enumerate(self.data):
    #         self.pair_news.append({"title": title, "body": body})


    def process_and_save(self, data):
        self.pair_news = unpack_pair_generator(data)
        count_features = {}
        for part in self.parts:
            for n, gram in enumerate(self.ngrams):
                count_features[part + "_" + gram] = \
                    list(get_ngram(n, list(map(lambda x: x[part], self.pair_news))))
                count_features["count_" + part + "_" + gram] = \
                    self.get_article_part_count(list(map(lambda x: x[part], self.pair_news)), n)
                count_features["count_unique_" + part + "_" + gram] = \
                    self.get_article_part_count(list(map(lambda x: set(x[part]), self.pair_news)), n)
                count_features["ratio_of_unique_" + part + "_" + gram] = \
                    list(map(lambda x, y: division(x, y),
                             count_features["count_unique_" + part + "_" + gram],
                             count_features["count_" + part + "_" + gram]))

        for gram in self.ngrams:
            count_features["count_of_title_"+gram+"_in_body"] = \
                list(map(lambda x, y:
                         sum([1. for word in x if word in set(y)]),
                         count_features["title_"+gram], count_features["body_"+gram]))
            count_features["ratio_of_title_" + gram + "_in_body"] = \
                list(map(division,
                         count_features["count_of_title_"+gram+"_in_body"],
                         count_features["count_title_"+gram]))

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
        X = self.count_features_df.drop("label", axis=1)
        y = self.count_features_df["label"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)
        clf = LogisticRegression(solver='saga', multi_class = 'multinomial')
        model = clf.fit(X_train, y_train)
        result = clf.predict(X_test[0])
        print(result)
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

    def read_trainset(self):
        with open("train_count_feature.json", mode="r") as f:
            cf = json.load(f)
            return cf

    def read_testset(self):
        with open("test_count_feature.json", mode="r") as f:
            cf = json.load(f)
            return cf


# if __name__ == '__main__':
#     cf = CountFeatureGenerator()
#     cf.read()