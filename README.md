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
- `get_ngram(n, sentence)` return n gram
- `tsne_similar_word_plot(model, word)` feed in model and a word, plot tsne of similar words.    

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


## Authors

* **Jun Lin**
* **Glenna Tremblay-Taylor**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
