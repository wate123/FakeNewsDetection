from nltk import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd, os


class SentimentFeatureGenerator(object):
    """
    Generate Sentiment feature and write into csv file.
    """
    def __init__(self, out_file_path):
        """
        Initializer that constructs Sentiment analyzer
        """
        # self.data = data
        self.pair_news = []
        # self.sentiment_feature_df = pd.DataFrame()
        self.sid = SentimentIntensityAnalyzer()
        self.out_file_path = out_file_path
        self.datasetName = out_file_path.split("_")[0]

    def compute_sentiment(self, sentences):
        """
        Feed in title sentences.
        Compute polarity scores and average them
        :param sentences: title in sentence format
        :return: average of polarity scores
        """
        results = [self.sid.polarity_scores(sentence)for sentence in sentences]
        return pd.DataFrame(results).mean()

    def process_and_save(self):
        """
        Generate title and body polarity score for each news
        :return:
        """
        print("Generate Title Sentiment Features")
        self.pair_news = pd.read_csv(self.out_file_path)

        title_sentiment_feature_df = pd.DataFrame()
        body_sentiment_feature_df = pd.DataFrame()

        # calculate polarity score for tokenized title and store result in data frame
        title_sentiment_feature_df["title_sent"] = self.pair_news["title"].apply(lambda x: sent_tokenize(str(x)))
        title_sentiment_feature_df = pd.concat(
            [title_sentiment_feature_df,
             title_sentiment_feature_df["title_sent"].apply(lambda x: self.compute_sentiment(x))], axis=1)

        # label whether positive, negative or neutral
        title_sentiment_feature_df.rename(
            columns={'compound': 'title_compound', 'neg': 'title_neg', 'neu': 'title_neu', 'pos': 'title_pos'},
            inplace=True)
        title_sentiment_feature_df["label"] = self.pair_news["label"].tolist()
        try:
            os.makedirs("./Features/"+self.datasetName)
        except OSError:
            pass
        # store results into csv
        title_sentiment_feature_df.drop("title_sent", axis=1).to_csv("./Features/"+self.datasetName+"/title_sentiment_feature.csv", index=False)
        print("Article title Done!")
        print("Save into title_sentiment_feature.csv")
        print()

        print("Generate Body Sentiment Features")

        # calculate polarity score for tokenized body and store results in data frame
        body_sentiment_feature_df["body_sent"] = self.pair_news["body"].apply(lambda x: sent_tokenize(str(x)))
        body_sentiment_feature_df = pd.concat(
            [body_sentiment_feature_df,
             body_sentiment_feature_df["body_sent"]
                 .apply(lambda x: self.compute_sentiment(x))], axis=1)

        # label whether positive, negative or neutral
        body_sentiment_feature_df.rename(
            columns={'compound': 'body_compound', 'neg': 'body_neg', 'neu': 'body_neu', 'pos': 'body_pos'},
            inplace=True)
        body_sentiment_feature_df["label"] = self.pair_news["label"].tolist()


        # store results into csv
        body_sentiment_feature_df.drop("body_sent", axis=1).to_csv("./Features/"+self.datasetName+"/body_sentiment_feature.csv", index=False)
        print("Article body Done!")
        print("Save into body_sentiment_feature.csv")
        return {"Tile Sentiment Feature Path": "./Features/"+self.datasetName+"/title_sentiment_feature.csv",
                "Body Sentiment Feature Path": "./Features/" + self.datasetName + "/body_sentiment_feature.csv"}

    def read(self):
        """
        Function that reads directly from files and merges the polarity results of title and body
        :return: merged sentiment results
        """
        title_sen_feat = pd.read_csv("./Features/"+self.datasetName+'/title_sentiment_feature.csv', index_col=False).drop("label", axis=1)
        body_sen_feat = pd.read_csv("./Features/"+self.datasetName+'/body_sentiment_feature.csv', index_col=False).drop("label", axis=1)

        return pd.merge(title_sen_feat, body_sen_feat, left_index=True, right_index=True)
