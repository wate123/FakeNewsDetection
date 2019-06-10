from os import walk
from os.path import join
import json, string, re
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric, remove_stopwords, stem_text
# Memory saving loading words


class NewsContent(object):
    def __init__(self, dirname, site, news_type):
        self.dirname = dirname
        self.site = site
        self.news_type = news_type
        self.list_news_files = self.get_list_news_files('json')
        # print(self.list_news_files)

    def __iter__(self):
        for file_path in self.list_news_files:
            with open(file_path, 'r') as f:
                CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric, remove_stopwords]
                text = json.load(f)['text']
                words = preprocess_string(text, filters=CUSTOM_FILTERS)
                print(words)
                yield words

    '''
    Return files path iterator you want in the provided directory
    @:param directory root direction you want to search
    '''

    def get_list_news_files(self, file_ext):
        list_news_files = []
        site_folder = join(self.dirname, self.site)
        news_path = join(site_folder, self.news_type)
        for root, dirs, files in walk(news_path):
            for f in files:
                if f.endswith("." + file_ext) and not dirs:
                    list_news_files.append(join(root, f))
        return iter(list_news_files)
