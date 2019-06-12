import json
import re
from os import walk
from os.path import join
import nltk

import matplotlib.pyplot as plt
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces, \
    strip_numeric, remove_stopwords
# % matplotlib inline
from sklearn.manifold import TSNE


# Memory saving loading words
#global variables 
english_stemmer = nltk.stem.SnowballStemmer('english')
stopwords = set(nltk.corpus.stopwords.words('english'))
token_pattern = r"(?u)\b\w\w+\b"

class NewsContent(object):
    def __init__(self, dirname, site, news_type, feature_type):
        self.dirname = dirname
        self.site = site
        self.news_type = news_type
        self.feature_type = feature_type
        self.list_news_files = self.get_list_news_files('json')

     #for stemming data 
     def stem_tokens(tokens, stemmer):
        stemmed= []
        for token in tokens:
            stemmed.append(stemmer.stem(token))
        return stemmed
    
    #for preprocessing data as tokens still need to modify 
    def preprocess(line, token_pattern=token_pattern, exclude_stopword=True,stem=True):
            token_pattern = re.compile(token_pattern, flags = re.UNICODE)
            tokens = [x.lower() for x in token_pattern.findall(line)]
            tokens_stemmed = tokens
            
            if stem:
                tokens_stemmed = NewsContent.stem_tokens(tokens, english_stemmer)
            if exclude_stopword:
                tokens_stemmed = [x for x in tokens_stemmed if x not in stopwords]
            return tokens_stemmed 
    
    def remove_emoji(self, text):
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

        # removing any identified emojis from text
        words = emoji_pattern.sub(r'', text)
        return words

    def __iter__(self):
        for file_path in self.list_news_files:
            with open(file_path, 'r') as f:
                doc = json.load(f)
                assert self.feature_type in doc.keys(), "feature not in the document: " + file_path
                # without stemming
                CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces,
                                  strip_numeric, remove_stopwords]
                feature = doc[self.feature_type]
                # self.title = json.load(f)['text']
                words = self.remove_emoji(feature)
                words = NewsContent.preprocess(words)
                #using alternative preprocessing function 
                #words = preprocess_string(words, filters=CUSTOM_FILTERS)
                yield words

    '''
    Return files path iterator you want in the provided directory
    @:param directory root direction you want to search
    '''

    def get_list_news_files(self, file_ext):
        list_news_files = []
        site_folder = join(self.dirname, self.site)
        news_path = join(site_folder, self.news_type)
        exclude = ["tweets", "retweets", "user_profile", "user_timeline_tweets", "user_followers", "user_following"]
        for root, dirs, files in walk(news_path, topdown=True):
            dirs[:] = [d for d in dirs if d not in exclude]
            # print(dirs)
            for f in files:
                if f.endswith("." + file_ext) and len(dirs) == 0:
                    list_news_files.append(join(root, f))
        return list_news_files

    # def get_title(self):


def tsne_similar_word_plot(model, word):
    "Creates and TSNE model and plots it"
    labels = [word]
    tokens = []

    for word, _ in model.similar_by_word(word):
        tokens.append(model[word])
        labels.append(word)

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()
