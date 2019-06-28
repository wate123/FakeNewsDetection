import json
import re
from os import walk
from os.path import join, exists
import nltk
from gensim.models.phrases import Phrases, Phraser
# from spellchecker import SpellChecker
from sklearn.model_selection import learning_curve

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# % matplotlib inline
from sklearn.manifold import TSNE

# global variables for preprocessing text data

english_stemmer = nltk.stem.SnowballStemmer('english')
stopwords = set(nltk.corpus.stopwords.words('english'))
token_pattern = r"(?u)\b\w\w+\b"


def stem_tokens(tokens, stemmer):
    """
    Function for tokens for stemming
    """
    stemmed = []
    # going through tokens stem where needed
    for token in tokens:
        stemmed.append(stemmer.stem(token))
    return stemmed


def preprocess(line, token_pattern=token_pattern, exclude_num=True, exclude_stopword=True, stem=True):
    """
    function to preprocess data by tokenizing, exclude: numbers, punctuation, stopwords, and stemming
    """
    # using regex for token patterns
    token_pattern = re.compile(token_pattern, flags=re.UNICODE)

    # tokenizing and making letters lowercase from data
    tokens = [remove_emoji(x.lower()) for x in token_pattern.findall(str(line))]
    tokens_stemmed = tokens

    # set to true so text is preprocessed
    if stem:
        # calls stemming function
        tokens_stemmed = stem_tokens(tokens, english_stemmer)

    if exclude_num:
        # removes numbers
        tokens_stemmed = [x for x in tokens_stemmed if not x.isdigit()]

    if exclude_stopword:
        # removes stopwords
        tokens_stemmed = [x for x in tokens_stemmed if x not in stopwords]
    return tokens_stemmed

# def spell_correction(text):
#
#     spell = SpellChecker()
#     correct = []
#
#     #misspelled = spell.unknown('text')
#     misspelled = spell.unknown(text)
#
#
#     for word in misspelled:
#         correct = spell.correction(word)
#
#         #print(spell.candidates(word))
#         #print(correct)
#     return correct

def remove_emoji(text):
    """function to remove emojis"""
    # using regex to identify all emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u'\U00010000-\U0010ffff'
                               u"\u200d"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\u3030"
                               u"\ufe0f"
                               "]+", flags=re.UNICODE)

    #removes all specified emoji found in text
    words = emoji_pattern.sub(r'', text)
    return words


class NewsContent(object):
    """
    Class to grab news data from fakenewsnet_dataset folder
    """

    def __init__(self, dirname, sites, news_types):
        """Initializer function containing directory, site and news type"""
        self.dirname = dirname
        self.sites = []
        self.news_types = []
        if type(sites) == str:
            self.sites.append(sites)
        if type(news_types) == str:
            self.news_types.append(news_types)
        else:
            self.sites = sites
            self.news_types = news_types
        self.list_news_path = list(self.get_list_news_files())
        # self.list_news_path = Parallel(n_jobs=-1)(delayed(list(self.get_list_news_files())))
        # self.feature_type = feature_type


    # def __iter__(self):
    #     for file_path in self.get_list_news_files():
    #         with open(file_path, 'r') as f:
    #             doc = json.load(f)
    #             assert self.feature_type in doc.keys(), "feature not in the document: " + file_path
    #             # without stemming
    #             CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces,
    #                               strip_numeric, remove_stopwords]
    #             # title = doc['title']
    #             # body = doc['text']
    #             feature = doc[self.feature_type]
    #             # self.title = json.load(f)['text']
    #             # title_words = preprocess(remove_emoji(title))
    #             # body_words = preprocess(remove_emoji(body))
    #             words = preprocess(feature)
    #             #using alternative preprocessing function
    #             #words = preprocess_string(words, filters=CUSTOM_FILTERS)
    #             yield words

    def get_features(self, feature_type="all"):
        """
        generator function to get specific features from news content.
        @:param feature_type  default yield the concatenation of title and body with preprocessing one by one.
                = pair  yield tile and body with preprocessing in pair form one by one.
                = title or body  yield tile or body with preprocessing one by one.

        """
        # if exists(path="data.csv"):
        #     return pd.read_csv("data.csv")
        # else:
        # reading through directory
        for file_path in self.list_news_path:
            with open(file_path, 'r') as f:

                # open document to read and assign to doc
                doc = json.load(f)
                # skip the empty title or body
                if doc['title'] == "" or doc['text'] == "":
                    pass
                else:
                    # to extract all data from news content
                    if feature_type == "all":
                        news = doc['title'] + doc['text']

                        # preprocesses news content
                        words = preprocess(news)
                        yield words

                    # to extract title and text as a pair
                    elif feature_type == "pair":
                        title = preprocess(doc["title"])
                        body = preprocess(doc['text'])
                        yield title, body
                        # if not title or not body:
                        #     pass
                        # else:
                        #     yield title, body

                    # else you only need either title or body
                    else:
                        assert feature_type in doc.keys(), "feature not in the document: " + file_path
                        # without stemming
                        # CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces,
                        #                   strip_numeric, remove_stopwords]

                        feature = doc[feature_type]
                        words = preprocess(feature)
                        # using alternative preprocessing function
                        # words = preprocess_string(words, filters=CUSTOM_FILTERS)
                        yield words

    def save_in_sentence_form(self):
        """Generate a json file for each news that contains their tile, body, and label"""
        big_dict = []

        # iterating through directories
        for file_path in self.list_news_path:
            with open(file_path, 'r') as f:
                # loading news content into labelled sections
                doc = json.load(f)
                if doc["title"] == "" or doc["text"] == "":
                    pass
                else:
                    big_dict.append(
                        {"title": remove_emoji(doc["title"]),
                         "body": remove_emoji(doc["text"]),
                         "label": str(file_path.split('/')[-3])})
        # write contents of dictionary to file
        print(len(big_dict))
        pd.DataFrame(big_dict).to_csv("data.csv", index=False)
        # with open("data.json", 'w+') as file:
        #     json.dump(big_dict, file)

    def get_list_news_files(self):
        """Return files path iterator of news"""
        # list_news_files = []
        for site in self.sites:
            for news_type in self.news_types:

                # accessing files through directories
                site_folder = join(self.dirname, site)
                news_path = join(site_folder, news_type)

                # only obtaining the news articles at this time
                exclude = ["tweets", "retweets", "user_profile", "user_timeline_tweets", "user_followers",
                           "user_following"]

                # iterating through directories only focusing on ones containing the news content
                for root, dirs, files in walk(news_path, topdown=True):
                    dirs[:] = [d for d in dirs if d not in exclude]

                    # collecting all articles
                    for f in files:
                        if f.endswith(".json") and len(dirs) == 0:
                            yield join(root, f)
                            # list_news_files.append(join(root, f))
        # print(len(list_news_files))
        # return list_news_files

        def get_list_twitter_files(self):
            """Return files path iterator of news"""
            list_twitter_files = []
            for site in self.sites:
                for news_type in self.news_types:

                    # accessing files through directories
                    site_folder = join(self.dirname, site)
                    news_path = join(site_folder, news_type)

                    # only obtaining the tweets/retweets at this time
                    exclude = ["news", "user_profile", "user_timeline_tweets", "user_followers",
                               "user_following"]

                    # iterating through directories only focusing on ones containing the news content
                    for root, dirs, files in walk(news_path, topdown=True):
                        dirs[:] = [d for d in dirs if d not in exclude]

                        # collecting all articles
                        for f in files:
                            if f.endswith(".json") and len(dirs) == 0:
                                yield join(root, f)
                                list_twitter_files.append(join(root, f))
            print(len(list_twitter_files))
            # return list_news_files


def get_ngram(n, sentence):
    """
    Function to get n grams to examine relationship between words in the news content
    """
    if n == 1:
        return list(sentence)
    
    # create phrases model to find words and ngrams that occur at least once
    ngram = Phraser(Phrases(sentence, min_count=1, threshold=1))

    # for bigrams and higher grams
    for i in range(2,n):
        ngram = Phraser(Phrases(ngram[sentence], min_count=1, threshold=1))
    return ngram[sentence]


def tsne_similar_word_plot(model, word):
    """
    Function to visualize relationships between filtered vocab from news content with TSNE
    """
    labels = [word]
    tokens = []

    # iterating through data to find similar words to given input
    for word, _ in model.similar_by_word(word):
        tokens.append(model[word])
        labels.append(word)

    # create TSNE model with 2 dimensions, using principal component analysis    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    
    # fits tokens into embedded space and returns transformation of tokens
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []

    # assigning data to appropiate axes
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    # setting up figure size
    plt.figure(figsize=(16, 16))

    # plotting and labeling data
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()


def division(x, y, val = 0.0):
    """division function used to divide two number"""
    if y != 0.0:
        val = float(x)/y
    return val

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
    # return plt

# def unpack_pair_generator(data):
#     pairs = []
#     try:
#         for title, body in data:
#             pairs.append({"title": title, "body": body})
#     except TypeError:
#         pass
#     return pairs

# def train_test_split(data):
#     train_set = {}
#     test_set = {}
#     for feature in self.count_features.keys():
#         train, test = train_test_split(self.count_features[feature], test_size=0.2)
#         train_set[feature] = train
#         test_set[feature] = test
