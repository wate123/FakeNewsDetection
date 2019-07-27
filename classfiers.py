from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.feature_selection import chi2, SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV as GCV, ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn import metrics
import time, datetime


def ada_boost(gcv, class_weight, cv=1):
    grid_C = [0.1 * i for i in range(1, 21)]
    grid_N = [10 * i for i in range(1, 21)]
    if class_weight is not None:
        DT_para = DTC(max_depth=3, criterion="entropy", random_state=1, class_weight=class_weight)
    else:
        DT_para = DTC(max_depth=3, criterion="entropy", random_state=1)
    parameters = {'base_estimator': [DT_para],
                  'n_estimators': grid_N, 'learning_rate': grid_C, 'algorithm': ['SAMME'], 'random_state': [1]}
    if not gcv:
        print("Start AdaBoost training")
        print("Start " + str(datetime.datetime.fromtimestamp(time.time())))
        clf = AdaBoostClassifier(algorithm='SAMME', base_estimator=DTC(class_weight='balanced', criterion='entropy',
                                                                       max_depth=None, max_features=None,
                                                                       max_leaf_nodes=None, min_impurity_decrease=0.0,
                                                                       min_impurity_split=None, min_samples_leaf=1,
                                                                       min_samples_split=2, min_weight_fraction_leaf=0.0,
                                                                       presort=False, random_state=1, splitter='best'),
                                 learning_rate=1.8, n_estimators=10, random_state=1)
    else:
        print("Start AdaBoost hyperperameter tuning")
        print("Start " + str(datetime.datetime.fromtimestamp(time.time())))

        if cv == 1:
            # tricks to do without cross validation
            clf = GCV(AdaBoostClassifier(), parameters, cv=ShuffleSplit(test_size=0.20, n_splits=1), n_jobs=40)
        else:
            clf = GCV(AdaBoostClassifier(), parameters, cv=cv, n_jobs=40)
    return clf


def knn(gcv, class_weight, cv=1):
    print("No Class_weight")
    grid_K = [i for i in range(1, 11)]
    parameters = {'n_neighbors': grid_K, 'weights': ['uniform'], 'algorithm': ('ball_tree', 'kd_tree'),
                  'leaf_size': [30], 'p': [2], 'metric': ['minkowski'], 'metric_params': [None]}
    if not gcv:
        print("Start KNN training")
        print("Start " + str(datetime.datetime.fromtimestamp(time.time())))
        clf = KNeighborsClassifier(algorithm='kd_tree', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=40,
                                   n_neighbors=10, p=2, weights='uniform')
    else:
        print("Start KNN hyperperameter tuning")
        print("Start " + str(datetime.datetime.fromtimestamp(time.time())))

        if cv == 1:
            # tricks to do without cross validation
            clf = GCV(KNeighborsClassifier(), parameters, cv=ShuffleSplit(test_size=0.20, n_splits=1), n_jobs=40)
        else:
            clf = GCV(KNeighborsClassifier(), parameters, cv=cv, n_jobs=40)
    return clf


def dt(gcv, class_weight, cv=1):
    parameters = {'criterion': ('gini', 'entropy'), 'splitter': ['best'], 'max_depth': [None], \
                  'min_samples_split': [2], 'min_samples_leaf': [1], 'min_weight_fraction_leaf': [0.0], \
                  'max_features': [None], 'random_state': [1], 'max_leaf_nodes': [None], \
                  'min_impurity_decrease': [0.0], 'min_impurity_split': [None],
                  'presort': [False]}
    if class_weight is not None:
        parameters["class_weight"] = [class_weight]
    if not gcv:
        print("Start Decision Tree training")
        print("Start " + str(datetime.datetime.fromtimestamp(time.time())))
        clf = DTC(class_weight='balanced', criterion='entropy', max_depth=None, max_features=None, max_leaf_nodes=None,
                  min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2,
                  min_weight_fraction_leaf=0.0, presort=False, random_state=1, splitter='best')
    else:
        print("Start Decision Tree hyperperameter tuning")
        print("Start " + str(datetime.datetime.fromtimestamp(time.time())))

        if cv == 1:
            # tricks to do without cross validation
            clf = GCV(DTC(), parameters, cv=ShuffleSplit(test_size=0.20, n_splits=1), n_jobs=40)
        else:
            clf = GCV(DTC(), parameters, cv=cv, n_jobs=40)
    return clf


def svm(gcv, class_weight, cv=1):
    grid_C = [0.5 * i for i in range(1, 21)]
    parameters = {'C': grid_C, 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'degree': [1, 2, 3],
                  'gamma': ['auto'], 'coef0': [0.0], 'shrinking': [True], 'probability': [False],
                  'tol': [5e-4], 'cache_size': [200], 'verbose': [False],
                  'max_iter': [30000], "class_weight": ['balanced'], 'decision_function_shape': ['ovr'], 'random_state': [1], }
    if class_weight is not None:
        parameters["class_weight"] = [class_weight]
    if not gcv:
        print("Start SVM training")
        print("Start " + str(datetime.datetime.fromtimestamp(time.time())))

        clf = SVC(C= 2.5, cache_size= 200, class_weight= 'balanced', coef0= 0.0,
                  decision_function_shape= 'ovr', degree= 1, gamma= 'auto', kernel= 'linear',
                  max_iter= 30000, probability= False, random_state= 1, shrinking= True,
                  tol= 0.0005, verbose= False)
    else:
        print("Start SVM hyperperameter tuning")
        print("Start " + str(datetime.datetime.fromtimestamp(time.time())))

        if cv == 1:
            # tricks to do without cross validation
            clf = GCV(SVC(), parameters, cv=ShuffleSplit(test_size=0.20, n_splits=1), n_jobs=40)
        else:
            clf = GCV(SVC(), parameters, cv=cv, n_jobs=40)
    return clf


def random_forest(gcv, class_weight, cv=1):
    # parameters = {'bootstrap': [True, False],
    #               'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    #               'max_features': ['auto', 'sqrt'],
    #               'min_samples_leaf': [1, 2, 4],
    #               'min_samples_split': [2, 5, 10],
    #               'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
    grid_N = [10 * i for i in range(5, 31)]
    parameters = {'n_estimators': grid_N, 'criterion': ('entropy', 'gini'), 'max_depth': [None],
                  'min_samples_split': [2], 'min_samples_leaf': [1], 'min_weight_fraction_leaf': [0.0],
                  'max_features': ['auto'], 'max_leaf_nodes': [None],
                  'min_impurity_decrease': [0.0], 'min_impurity_split': [None], 'bootstrap': [True],
                  'oob_score': [False], 'random_state': [1], 'verbose': [0],
                  'warm_start': [False]}
    if class_weight is not None:
        parameters["class_weight"] = [class_weight]
    if not gcv:
        print("Start Random Forest training")
        print("Start " + str(datetime.datetime.fromtimestamp(time.time())))
        clf = RandomForestClassifier(class_weight=class_weight, bootstrap=True, criterion='gini', max_depth=None,
                                     max_features='auto',
                                     max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                                     min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0,
                                     n_estimators=110, n_jobs=40, oob_score=False, random_state=1, verbose=0,
                                     warm_start=False)
    else:
        print("Start Random Forest hyperperameter tuning")
        print("Start " + str(datetime.datetime.fromtimestamp(time.time())))

        if cv == 1:
            # tricks to do without cross validation
            clf = GCV(RandomForestClassifier(), parameters, cv=ShuffleSplit(test_size=0.20, n_splits=1), n_jobs=40)
        else:
            clf = GCV(RandomForestClassifier(), parameters, cv=cv, n_jobs=40)
    return clf


def xgboost(gcv, class_weight, cv=1):
    grid_R = [0.1 * i for i in range(1, 10)]
    grid_N = [10 * i for i in range(1, 21)]
    parameters = {'max_depth': [3, 6], 'learning_rate': grid_R, 'n_estimators': grid_N,
                  'objective': ['multi:softmax', 'multi:softprob'], 'booster': ['gbtree', 'dart'],
                  'subsample': [0.7, 0.8, 0.9, 1.], 'random_state': [1], "num_class": [2, 4, 6]
                  }
    # "scale_pos_weight": [0.31]
    if class_weight is not None:
        parameters["class_weight"] = [class_weight]
    if not gcv:
        print("Start XGBoost training")
        print("Start " + str(datetime.datetime.fromtimestamp(time.time())))
        clf = xgb.XGBClassifier(booster='dart', learning_rate=0.30000000000000004, max_depth=6, n_estimators=120, num_class=6,
                                objective='multi:softmax', random_state=1, subsample=0.8, n_jobs=40, verbosity=0,
                                scale_pos_weight=0.31)
    else:
        print("Start XGBoost hyperperameter tuning")
        print("Start " + str(datetime.datetime.fromtimestamp(time.time())))
        if cv == 1:
            # trick to do this without cross validation
            clf = GCV(xgb.XGBClassifier(), parameters, cv=ShuffleSplit(test_size=0.20, n_splits=1), n_jobs=40)
        else:
            clf = GCV(xgb.XGBClassifier(), parameters, cv=cv, n_jobs=40)
    return clf


def logistic_reg(gcv, class_weight, cv=1):
    grid_C = [0.5 * i for i in range(1, 21)]
    parameters = {"tol": [5e-4], "C": grid_C, "random_state": [1],
                  "solver": ["newton-cg", "sag", "saga", "lbfgs"],
                  "max_iter": [4000], "multi_class": ["multinomial", "ovr", "auto"]}
    if class_weight is not None:
        parameters["class_weight"] = [class_weight]
    if not gcv:
        print("Start Logistic Regression training")
        print("Start " + str(datetime.datetime.fromtimestamp(time.time())))
        clf = LogisticRegression(tol=0.0005, C=7.0, max_iter=4000, multi_class='multinomial', random_state=1, solver='sag',
                                 class_weight=None, n_jobs=40)
    else:
        print("Start Logistic Regression hyperperameter tuning")
        print("Start " + str(datetime.datetime.fromtimestamp(time.time())))
        if cv == 1:
            # trick to do this without cross validation
            clf = GCV(LogisticRegression(), parameters, cv=ShuffleSplit(test_size=0.20, n_splits=1), n_jobs=40)
        else:
            clf = GCV(LogisticRegression(), parameters, cv=cv, n_jobs=40)
    return clf
