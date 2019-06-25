from utils import get_ngram
# from FeatureGenerator import FeatureGenerator
from collections import defaultdict
from utils import division
import json
from nltk import sent_tokenize
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, utils
from sklearn.preprocessing import StandardScaler, scale
from sklearn.model_selection import GridSearchCV as GCV, validation_curve, train_test_split, ShuffleSplit, learning_curve
import matplotlib.pyplot as plt
from joblib import dump, load

def get_article_part_count(part, ngram=1):
    """
    get ngram count of article title or body
    :param part: title
    :param ngram:
    :return:
    """
    return [len(t) for t in get_ngram(ngram, part)]


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
        # a list of title and body key value pairs
        with open("data.json", mode="r") as f:
            self.pair_news = json.load(f)

        count_features = {}
        ngrams = {}

        # generate count, unique count, and ratio of unique count and count (unique count / count)
        # of title, body, and uni to tri gram
        for part in self.parts:
            for n, gram in enumerate(self.ngrams):
                ngrams[part + "_" + gram] = \
                    list(get_ngram(n, list(map(lambda x: x[part], self.pair_news))))
                count_features["count_" + part + "_" + gram] = \
                    get_article_part_count(list(map(lambda x: x[part], self.pair_news)), n)
                count_features["count_unique_" + part + "_" + gram] = \
                    get_article_part_count(list(map(lambda x: set(x[part]), self.pair_news)), n)
                count_features["ratio_of_unique_" + part + "_" + gram] = \
                    list(map(lambda x, y: division(x, y),
                             count_features["count_unique_" + part + "_" + gram],
                             count_features["count_" + part + "_" + gram]))
        # count of ngram title in body,
        # ratio of ngram title in body (count of ngram title in body / count of ngram title)
        for gram in self.ngrams:
            count_features["count_of_title_"+gram+"_in_body"] = \
                list(map(lambda x, y:
                         sum([1. for word in x if word in set(y)]),
                         ngrams["title_"+gram], ngrams["body_"+gram]))
            count_features["ratio_of_title_" + gram + "_in_body"] = \
                list(map(division,
                         count_features["count_of_title_"+gram+"_in_body"],
                         count_features["count_title_"+gram]))

        # get label of each news and count number of sentence in title and body
        # with open("data.json", mode="r") as f:
        #     data = json.load(f)
        label = []
        len_sent_title = []
        len_sent_body = []
        for news in self.pair_news:
            label.append(news["label"])
            len_sent_title.append(len(sent_tokenize(news["title"])))
            len_sent_body.append(len(sent_tokenize(news["body"])))
        count_features["label"] = label
        count_features["len_sent_title"] = len_sent_title
        count_features["len_sent_body"] = len_sent_body
            # count_features["label"] = [news["label"] for news in data]
            # count_features["len_sent_title"] = [len(sent_tokenize(news["title"])) for news in data]
            # count_features["len_sent_body"] = [len(sent_tokenize(body))for _, body in data]
        self.count_features_df = pd.DataFrame.from_dict(count_features)
        pd.DataFrame.to_csv(self.count_features_df, "count_feature.csv")

        # pd.DataFrame.to_csv("")
        # self.count_features["len_sent_title"] = [len(sent_tokenize(" ".join(news["title"]))) for news in self.pair_news]
        # self.count_features["len_sent_body"] = [len(sent_tokenize(" ".join(news["body"]))) for news in self.pair_news]

        # train_set = {}
        # test_set = {}
        # for feature in self.count_features.keys():
        #     train, test = train_test_split(self.count_features[feature], test_size=0.2)
        #     train_set[feature] = train
        #     test_set[feature] = test
        #
        # with open("train_count_feature.json", mode="w+") as f:
        #     json.dump(train_set, indent=4, fp=f)
        #
        # with open("test_count_feature.json", mode="w+") as f:
        #     json.dump(test_set, indent=4, fp=f)
        # with open("count_feature.json", mode="w+") as f:
        #     json.dump(self.count_features, indent=4, fp=f)


    def read(self, trained):
        """
        read directly from feature file and split train test set and make prediction using 20% test set
        """
        df = pd.read_csv('count_feature.csv', index_col=False)
        X = scale(df.drop("label", axis=1).values)
        print("shape: ", X.shape)
        # X = df.drop("label", axis=1).values
        y = df["label"].values
        X, y = utils.shuffle(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        grid_C = [0.5 * i for i in range(1, 21)]
        parameters = {"tol": [5e-4], "C": grid_C, "random_state":[1],
                      "solver": ["newton-cg", "sag", "saga", "lbfgs"],
                      "max_iter": [4000], "multi_class": ["multinomial", "ovr", "auto"]}

        clf = LogisticRegression(tol=0.0005, C=8.5, max_iter=4000, multi_class='multinomial', random_state=1, solver='newton-cg' )
        # if trained:
        #     clf = load('logreg_count_feature.joblib')
        # else:
        #     clf = GCV(LogisticRegression(), parameters, cv=10, n_jobs=-1)
        #     dump(clf, 'logreg_count_feature.joblib')

        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        tpfptnfn = metrics.confusion_matrix(y_test, y_predict)
        preRecF1 = metrics.classification_report(y_test, y_predict)
        # print(clf.best_params_)
        print(preRecF1)
        # plot_learning_curve(clf, "Learning curve", X_train, y_train, n_jobs=-1, cv=10)


        # clf = LogisticRegression(solver="saga")
        #
        # model = clf.fit(X_train, y_train)
        #
        # result = clf.predict(X_test)
        # score = metrics.accuracy_score(y_test, result)
        # precision = metrics.precision_score(y_test, result, ["fake", "real"], pos_label="real")
        # recall = metrics.recall_score(y_test, result, ["fake", "real"], pos_label="real")
        # f1 = metrics.f1_score(y_test, result, ["fake", "real"], pos_label="real")
        # print("accuracy: ", score)
        # print("precision: ", precision)
        # print("recall: ", recall)
        # print("f1: ", f1)

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


# if __name__ == '__main__':
#     cf = CountFeatureGenerator()
#     cf.read()