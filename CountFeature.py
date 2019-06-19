from utils import get_ngram
# from FeatureGenerator import FeatureGenerator
from collections import defaultdict
from utils import division
import json
from nltk import sent_tokenize
from utils import unpack_pair_generator
from FeatureGenerator import FeatureGenerator


class CountFeatureGenerator(object):
    # def __init__(self):
    def __init__(self, data, name='countFeatureGenerator' ):
        # super(CountFeatureGenerator, self).__init__(name)
        self.data = data
        self.pair_news = unpack_pair_generator(data)
        self.parts = ["title", "body"]
        self.ngrams = ["uni", "bi", "tri"]
        self.count_features = {}
        # self.unpack_pair_generator()

    def get_article_part_count(self, part, ngram=1):
        # return [len() for count in get_ngram(ngram, self.data)]
        # pair_dict = dict((title, body) for title, body in self.data)
        # print(pair_dict)
        return [len(t) for t in get_ngram(ngram, part)]

    # def unpack_pair_generator(self):
    #     for count, (title, body) in enumerate(self.data):
    #         self.pair_news.append({"title": title, "body": body})


    def process_and_save(self):
        for part in self.parts:
            for n, gram in enumerate(self.ngrams):
                self.count_features[part+"_"+gram] = list(get_ngram(n, list(map(lambda x: x[part], self.pair_news))))
                self.count_features["count_"+part+"_"+gram] = \
                    self.get_article_part_count(list(map(lambda x: x[part], self.pair_news)), n)
                self.count_features["count_unique_"+part+"_"+gram] = \
                    self.get_article_part_count(list(map(lambda x:
                                                         set(x[part]), self.pair_news)), n)
                self.count_features["ratio_of_unique_"+part+"_"+gram] = \
                    list(map(lambda x,y: division(x, y),
                        self.count_features["count_unique_" + part + "_" + gram],
                        self.count_features["count_"+part + "_" + gram]))

        for gram in self.ngrams:
            self.count_features["count_of_title_"+gram+"_in_body"] = \
                list(map(lambda x, y:
                         sum([1. for word in x if word in set(y)]),
                         self.count_features["title_"+gram], self.count_features["body_"+gram]))
            self.count_features["ratio_of_title_" + gram + "_in_body"] = \
                list(map(division,
                         self.count_features["count_of_title_"+gram+"_in_body"],
                         self.count_features["count_title_"+gram]))

        with open("data.json", mode="r") as f:
            data = json.load(f)
            self.count_features["len_sent_title"] = [len(sent_tokenize(title)) for title in data]
            self.count_features["len_sent_body"] = [len(sent_tokenize(body)) for _, body in data.items()]

        # self.count_features["len_sent_title"] = [len(sent_tokenize(" ".join(news["title"]))) for news in self.pair_news]
        # self.count_features["len_sent_body"] = [len(sent_tokenize(" ".join(news["body"]))) for news in self.pair_news]
        with open("count_feature.json", mode="w+") as f:
            json.dump(self.count_features, indent=4, fp=f)




# if __name__ == '__main__':
#     cf = CountFeatureGenerator()
#     cf.read()