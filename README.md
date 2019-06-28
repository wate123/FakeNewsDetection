# Fake News Detection

## Getting Started


### Prerequisites

- [FakeNewsNet Dataset](https://github.com/KaiDMML/FakeNewsNet)
- python 3
- gensim
- scikit-learn
- nltk
- matplotlib
- numpy, pandas
- imblearn


### Installing

```
pip install -r requirement.txt
```

### Run
Replace the first parameter to your fakenewsnet_dataset location.
````
NewsContent('../FakeNewsNet/code/fakenewsnet_dataset', ['politifact'], ['fake', 'real'])
````
````
python main.py
````

### Project Structure
**`main.py`**:
process data and generate feature. 

**`utils.py`**:   
- NewsContent Class
    - `get_features()` generator function that returns news title, body, or both preprocessed.
    - `save_in_sentence_form()` generate a json file of all news content with title, body, label key value pair.
    - `get_list_news_files()` generator function that yield each of news file path.
- `stem_tokens(tokens, stemmer)` stem tokens for preprocessing
- `preprocess(line, token_pattern=token_pattern, exclude_num=True, exclude_stopword=True, stem=True)` tokenize words and        preprocess 
- `remove_emoji(text)` remove emojis for preprocessing 
- `get_ngram(n, sentence)` return n gram
- `tsne_similar_word_plot(model, word)` feed in model and a word, plot tsne of similar words.  
- `division(x, y, val = 0.0)` to divide two numbers
- `plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5))`          generate plot of training and testing learning curve 

**`CountFeature.py`**  :    
- CountFeatureGenerator Class
    - `process_and_save()` takes title and body pair data and write count feature into csv file.
    - `read()` read the count feature csv file and make prediction
- `get_article_part_count` count ngram of title or body 

**`SentimentFeature.py`**: 
- SentimentFeatureGenerator Class
    - `compute_sentiment()` compute polarity score of each sentences in title or body and average them
    - `process_and_save()` takes title and body pair data and write polarity score feature of title and body into csv file.
    - `read()` read the title or body feature csv file and make prediction
    
**`Word2VecFeature.py`**: 
- Word2VecFeatureGenerator Class
    - `get_norm_vectors()` get normalized word vector
    - `get_title_body_cos_sim()` cosine similarity of title and body   
    - `process_and_save()` takes title and body pair data and write polarity score feature of title and body into csv file.
    - `read()` read the title or body feature csv file and make prediction     
    
**`SvdFeature.py`**:
- SvdFeature Class
    - `process_tfidf()` get tf-idf matrix
    - `process_and_save()` use SVD (or NMF) to reduce Tf-idf matrix and write into csv file
    - `read()` read the svd feature csv file and make prediction
    - `get_tfidf_scores()` to get vocab and their corresponding scores from tf-idf matrix 
    
**`Parameters.py`**:
To hold best parameters for various classifier models. 


## Authors

* **Jun Lin**
* **Glenna Tremblay-Taylor**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
