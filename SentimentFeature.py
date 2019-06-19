from nltk import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json
import statistics
from utils import unpack_pair_generator, division
from collections import defaultdict


class SentimentFeatureGenerator(object):

    def __init__(self):
        # self.data = data
        self.pair_news = []
        self.sentiment_feature = {}
        self.sid = SentimentIntensityAnalyzer()

    def compute_sentiment(self, sentences):
        results = [self.sid.polarity_scores(sentence) for sentence in sentences]
        return self.average_polarity_scores(results)

    def average_polarity_scores(self, scores):
        """Mean"""
        avg_scores = defaultdict(list)
        for score in scores:
            for key, val in score.items():
                avg_scores[key].append(val)
        for label in avg_scores:
            avg_scores[label] = statistics.mean(avg_scores[label])
        return dict(avg_scores)

    def process_and_save(self):
        print("Generate Sentiment Features")
        with open('data.json', mode="r") as f:
            data = json.load(f)
            self.pair_news = unpack_pair_generator((title, body) for title, body in data.items())
        self.sentiment_feature["title_polarity_score"] = \
            [self.compute_sentiment(sent_tokenize(news["title"])) for news in self.pair_news]
        self.sentiment_feature["body_polarity_score"] = \
            [self.compute_sentiment(sent_tokenize(news["body"])) for news in self.pair_news]
