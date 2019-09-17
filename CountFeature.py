from utils import get_ngram, division
from nltk import sent_tokenize
import pandas as pd
from utils import preprocess
import os

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
    def __init__(self, out_file_path, name='countFeatureGenerator'  ):
        """
        Initializer that constructs components of count feature
        """
        # super(CountFeatureGenerator, self).__init__(name)
        # self.data = data
        self.pair_news = pd.DataFrame()
        self.parts = ["title", "body"]
        self.ngrams = ["uni", "bi", "tri"]
        self.count_features_df = pd.DataFrame()
        self.out_file_path = out_file_path
        self.datasetName = out_file_path.split("_")[0]
        # self.data = data
        # self.unpack_pair_generator()

    def process_and_save(self):
        """
        Function that counts ngrams, unique count and ratio of the two
        """
        print("Generating Count Features")
        # a list of title and body key value pairs
        self.pair_news = pd.read_csv(self.out_file_path)

        # title_uni_list, body_uni_list = zip(*self.data)
        # self.pair_news["title"] = list(title_uni_list)
        # self.pair_news["body"] = list(body_uni_list)
        # self.pair_news["label"] = pd.read_csv("data.csv")["label"]
        ngrams = {}
        # print(max([len(part)for part in list(body_uni_list)]))
        # generate count, unique count, and ratio of unique count and count (unique count / count)
        # of title, body, and uni to tri gram
        for part in self.parts:
            #preprocess data
            unigram = self.pair_news[part].astype(str).apply(preprocess)
            # unigram = self.pair_news[part].values
            bigram = list(get_ngram(2, unigram))


            for n, gram in enumerate(self.ngrams, 1):
                ngrams[part + "_" + gram] = list(get_ngram(n, unigram))

                #store results in data frame
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

        # length of title and body of articles stored in data frame
        self.count_features_df["len_sent_title"] = self.pair_news["title"].astype(str).apply(lambda x: len(x), sent_tokenize)
        self.count_features_df["len_sent_body"] = self.pair_news["body"].astype(str).apply(lambda x: len(x), sent_tokenize)

        self.count_features_df["label"] = self.pair_news["label"]

        try:
            os.makedirs("./Features/"+self.datasetName)
        except OSError:
            pass
        # remove index to ensure model does not use during training or testing
        self.count_features_df.to_csv("./Features/"+self.datasetName+"/count_feature.csv", index=False)
        print("Done! save into count_feature.csv")
        return {"Count Feature Path": "./Features/"+self.datasetName+"/count_feature.csv"}

    def read(self):
        """
        read directly from feature file and split train test set and make prediction using 20% test set
        """
        return pd.read_csv("./Features/"+self.datasetName+'/count_feature.csv', index_col=False).drop("label", axis=1)

# if __name__ == '__main__':
#     cf = CountFeatureGenerator()
#     cf.read()