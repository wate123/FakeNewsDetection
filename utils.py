from os import walk
from os.path import join
import json, re
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

        
    def remove_emoji(text):
        #using regex to identify all emojis
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
    
        #removing any identified emojis from text 
        words = emoji_pattern.sub(r'', text)
        return words
    
    def __iter__(self):
        for file_path in self.list_news_files:
            with open(file_path, 'r') as f:
                CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric, remove_stopwords]
                text = json.load(f)['text']
                
                #removes emojis from text 
                words = NewsContent.remove_emoji(text)
                words = preprocess_string(words, filters=CUSTOM_FILTERS)
                # print(words)
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
        return iter(list_news_files)
