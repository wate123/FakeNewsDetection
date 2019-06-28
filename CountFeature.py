from utils import get_ngram, division
from nltk import sent_tokenize
import pandas as pd
from utils import preprocess

def get_article_part_count(part, unique=False):
    """
    get ngram count of article title or body
    :param part: title
    :param ngram:
    :return:
    """
    if unique:
        return [len(set(t)) for t in part]
    return [len(t) for t in part]


class CountFeatureGenerator(object):
    """
    Generate Count feature and write into csv file.
    """
    def __init__(self, name='countFeatureGenerator' ):
        # super(CountFeatureGenerator, self).__init__(name)
        # self.data = data
        self.pair_news = {}
        self.parts = ["title", "body"]
        self.ngrams = ["uni", "bi", "tri"]
        self.count_features_df = pd.DataFrame()
        # self.unpack_pair_generator()

    def process_and_save(self):
        print("Generating Count Features")
        # a list of title and body key value pairs
        self.pair_news = pd.read_csv("data.csv")

        ngrams = {}

        # generate count, unique count, and ratio of unique count and count (unique count / count)
        # of title, body, and uni to tri gram
        for part in self.parts:
            unigram = self.pair_news[part].astype(str).apply(preprocess)
            for n, gram in enumerate(self.ngrams):
                ngrams[part + "_" + gram] = list(get_ngram(n, unigram))
                self.count_features_df["count_" + part + "_" + gram] = get_article_part_count(ngrams[part + "_" + gram])
                self.count_features_df["count_unique_" + part + "_" + gram] = \
                    get_article_part_count(ngrams[part + "_" + gram], unique=True)
                self.count_features_df["ratio_of_unique_" + part + "_" + gram] = \
                    list(map(lambda x, y: division(x, y),
                             self.count_features_df["count_unique_" + part + "_" + gram],
                             self.count_features_df["count_" + part + "_" + gram]))
        # count of ngram title in body,
        # ratio of ngram title in body (count of ngram title in body / count of ngram title)
        for gram in self.ngrams:
            self.count_features_df["count_of_title_"+gram+"_in_body"] = \
                list(map(lambda x, y:
                         sum([1. for word in x if word in set(y)]),
                         ngrams["title_"+gram], ngrams["body_"+gram]))
            self.count_features_df["ratio_of_title_" + gram + "_in_body"] = \
                list(map(division,
                         self.count_features_df["count_of_title_"+gram+"_in_body"],
                         self.count_features_df["count_title_"+gram]))

        self.count_features_df["len_sent_title"] = self.pair_news["title"].astype(str).apply(lambda x: len(x), sent_tokenize)
        self.count_features_df["len_sent_body"] = self.pair_news["body"].astype(str).apply(lambda x: len(x), sent_tokenize)

        self.count_features_df["label"] = self.pair_news["label"]
        self.count_features_df.to_csv("count_feature.csv", index=False)
        print("Done! save into count_feature.csv")

    def read(self):
        """
        read directly from feature file and split train test set and make prediction using 20% test set
        """
        return pd.read_csv('count_feature.csv', index_col=False).drop("label", axis=1)

# if __name__ == '__main__':
#     cf = CountFeatureGenerator()
#     cf.read()