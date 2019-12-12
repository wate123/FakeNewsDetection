from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.model_selection import GridSearchCV as GCV, ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier as DTC
import time, datetime

# For ada boost, you have to tune the decision tree first and replace the parameters
# in order to achieve the same result we have.
def ada_boost(gcv, default_param,dataset, class_weight, seed, cv=1):
    grid_C = [0.1 * i for i in range(1, 21)]
    grid_N = [10 * i for i in range(1, 21)]
    # dt_parameters = {'class_weight': 'balanced', 'criterion': 'entropy', 'max_depth': None, 'max_features': None,
    #                  'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2,
    #                  'min_weight_fraction_leaf': 0.0, 'presort': False, 'random_state': 1, 'splitter': 'best'}
    polti_param = DTC(**{'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': None, 'max_features': None,
           'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2,
           'min_weight_fraction_leaf': 0.0, 'presort': False, 'random_state': 1, 'splitter': 'best'})
    gossi_param = DTC(**{'class_weight': 'balanced', 'criterion': 'entropy', 'max_depth': None, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'presort': False, 'random_state': 1, 'splitter': 'best'})
    combine_param = DTC(**{'class_weight': 'balanced', 'criterion': 'entropy', 'max_depth': None, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'presort': False, 'random_state': 1, 'splitter': 'best'})
    # if class_weight is not False:
    #
    #     DT_para = DTC(**dt_parameters)
    # else:
    #     DT_para = DTC(**dt_parameters)

    poli_parameters = {'base_estimator': [polti_param],
                  'n_estimators': grid_N, 'learning_rate': grid_C, 'algorithm': ['SAMME'], 'random_state': [seed]}
    gossi_parameters = {'base_estimator': [gossi_param],
                  'n_estimators': grid_N, 'learning_rate': grid_C, 'algorithm': ['SAMME'], 'random_state': [seed]}
    combine_parameters = {'base_estimator': [combine_param],
                  'n_estimators': grid_N, 'learning_rate': grid_C, 'algorithm': ['SAMME'], 'random_state': [seed]}
    if not gcv:
        print("Start AdaBoost training")
        print("Start " + str(datetime.datetime.fromtimestamp(time.time())))
        if default_param:
            if dataset == "politifact":
                clf = AdaBoostClassifier(algorithm='SAMME',
                                         base_estimator=polti_param,
                                         random_state=seed)
                return clf, "Ada Boost", []
            if dataset == "gossipcop":
                clf = AdaBoostClassifier(algorithm='SAMME',
                                         base_estimator=gossi_param,
                                         random_state=seed)
                return clf, "Ada Boost", []
            if dataset == "politifact-gossipcop":
                clf = AdaBoostClassifier(algorithm='SAMME',
                                         base_estimator=combine_param,
                                         random_state=seed)
                return clf, "Ada Boost", []
        elif dataset == 'politifact':
            # {'algorithm': 'SAMME',
            #  'base_estimator': DecisionTreeClassifier(class_weight='balanced', criterion='entropy',
            #                                           max_depth=3, max_features=None, max_leaf_nodes=None,
            #                                           min_impurity_decrease=0.0, min_impurity_split=None,
            #                                           min_samples_leaf=1, min_samples_split=2,
            #                                           min_weight_fraction_leaf=0.0, presort=False,
            #                                           random_state=1, splitter='best'), 'learning_rate': 1.5,
            #  'n_estimators': 20, 'random_state': 1}
            clf = AdaBoostClassifier(**{'algorithm': 'SAMME', 'base_estimator': polti_param, 'learning_rate': 1.6, 'n_estimators': 10, 'random_state': 1})
            return clf, "Ada Boost", poli_parameters
        elif dataset == 'gossipcop':
            # {'algorithm': 'SAMME',
            #  'base_estimator': DecisionTreeClassifier(class_weight='balanced', criterion='entropy',
            #                                           max_depth=3, max_features=None, max_leaf_nodes=None,
            #                                           min_impurity_decrease=0.0, min_impurity_split=None,
            #                                           min_samples_leaf=1, min_samples_split=2,
            #                                           min_weight_fraction_leaf=0.0, presort=False,
            #                                           random_state=1, splitter='best'),
            #  'learning_rate': 1.9000000000000001, 'n_estimators': 10, 'random_state': 1}
            clf = AdaBoostClassifier(**{'algorithm': 'SAMME', 'base_estimator': gossi_param, 'learning_rate': 2.0, 'n_estimators': 150, 'random_state': 1})
            return clf, "Ada Boost", gossi_parameters
        elif dataset == 'politifact-gossipcop':
            # {'algorithm': 'SAMME',
            #  'base_estimator': DecisionTreeClassifier(class_weight='balanced', criterion='entropy',
            #                                           max_depth=3, max_features=None, max_leaf_nodes=None,
            #                                           min_impurity_decrease=0.0, min_impurity_split=None,
            #                                           min_samples_leaf=1, min_samples_split=2,
            #                                           min_weight_fraction_leaf=0.0, presort=False,
            #                                           random_state=1, splitter='best'), 'learning_rate': 1.6,
            #  'n_estimators': 10, 'random_state': 1}
            clf = AdaBoostClassifier(**{'algorithm': 'SAMME', 'base_estimator': combine_param, 'learning_rate': 0.1, 'n_estimators': 90, 'random_state': 1})
            return clf, "Ada Boost", combine_parameters
    else:
        print("Start AdaBoost hyperperameter tuning")
        print("Start " + str(datetime.datetime.fromtimestamp(time.time())))

        if cv == 1:
            # tricks to do without cross validation
            if dataset == 'politifact':
                clf = GCV(AdaBoostClassifier(), poli_parameters, cv=ShuffleSplit(test_size=0.20, n_splits=1), n_jobs=40)
                return clf, "Ada Boost", poli_parameters
            if dataset == 'gossipcop':
                clf = GCV(AdaBoostClassifier(), gossi_parameters, cv=ShuffleSplit(test_size=0.20, n_splits=1), n_jobs=40)
                return clf, "Ada Boost", gossi_parameters
            if dataset == 'politifact-gossipcop':
                clf = GCV(AdaBoostClassifier(), combine_parameters, cv=ShuffleSplit(test_size=0.20, n_splits=1), n_jobs=40)
                return clf, "Ada Boost", combine_parameters
        else:
            if dataset == 'politifact':
                clf = GCV(AdaBoostClassifier(), poli_parameters, cv=cv, n_jobs=40)
                return clf, "Ada Boost", poli_parameters
            if dataset == 'gossipcop':
                clf = GCV(AdaBoostClassifier(), gossi_parameters, cv=cv, n_jobs=40)
                return clf, "Ada Boost", gossi_parameters
            if dataset == 'politifact-gossipcop':
                clf = GCV(AdaBoostClassifier(), combine_parameters, cv=cv, n_jobs=40)
                return clf, "Ada Boost", combine_parameters
    # return clf, "Ada Boost", parameters


def knn(gcv, default_param, dataset, class_weight, seed, cv=1):
    print("No Class_weight")
    grid_K = [i for i in range(1, 11)]
    parameters = {'n_neighbors': grid_K, 'weights': ['uniform'], 'algorithm': ('ball_tree', 'kd_tree'),
                  'leaf_size': [30], 'p': [2], 'metric': ['minkowski'], 'metric_params': [None]}
    if not gcv:
        print("Start KNN training")
        print("Start " + str(datetime.datetime.fromtimestamp(time.time())))
        if default_param:
            clf = KNeighborsClassifier(n_jobs=40, weights='uniform', leaf_size=30, p=2, metric='minkowski', metric_params=None)
        elif dataset == 'politifact':
            # {'algorithm': 'ball_tree', 'leaf_size': 30, 'metric': 'minkowski', 'metric_params': None, 'n_neighbors': 6, 'p': 2, 'weights': 'uniform'}
            clf = KNeighborsClassifier(n_jobs=40, **{'algorithm': 'ball_tree', 'leaf_size': 30, 'metric': 'minkowski', 'metric_params': None, 'n_neighbors': 6, 'p': 2, 'weights': 'uniform'})
        elif dataset == 'gossipcop':
            # {'algorithm': 'ball_tree', 'leaf_size': 30, 'metric': 'minkowski', 'metric_params': None, 'n_neighbors': 7,
            #  'p': 2, 'weights': 'uniform'}
            clf = KNeighborsClassifier(n_jobs=40, **{'algorithm': 'ball_tree', 'leaf_size': 30, 'metric': 'minkowski', 'metric_params': None, 'n_neighbors': 7, 'p': 2, 'weights': 'uniform'})
        elif dataset == 'politifact-gossipcop':
            # {'algorithm': 'ball_tree', 'leaf_size': 30, 'metric': 'minkowski', 'metric_params': None, 'n_neighbors': 9,
            #  'p': 2, 'weights': 'uniform'}
            clf = KNeighborsClassifier(n_jobs=40, **{'algorithm': 'ball_tree', 'leaf_size': 30, 'metric': 'minkowski', 'metric_params': None, 'n_neighbors': 9, 'p': 2, 'weights': 'uniform'})
    else:
        print("Start KNN hyperperameter tuning")
        print("Start " + str(datetime.datetime.fromtimestamp(time.time())))

        if cv == 1:
            # tricks to do without cross validation
            clf = GCV(KNeighborsClassifier(), parameters, cv=ShuffleSplit(test_size=0.20, n_splits=1), n_jobs=40)
        else:
            clf = GCV(KNeighborsClassifier(), parameters, cv=cv, n_jobs=40)
    return clf, "K nearest neighbor", parameters


def dt(gcv, default_param, dataset, class_weight, seed, cv=1):
    parameters = {'criterion': ('gini', 'entropy'), 'splitter': ['best'], 'max_depth': [None], \
                  'min_samples_split': [2], 'min_samples_leaf': [1], 'min_weight_fraction_leaf': [0.0], \
                  'max_features': [None], 'random_state': [seed], 'max_leaf_nodes': [None], \
                  'min_impurity_decrease': [0.0], 'presort': [False]}
    if class_weight is not False:
        parameters["class_weight"] = [class_weight]
    if not gcv:
        print("Start Decision Tree training")
        print("Start " + str(datetime.datetime.fromtimestamp(time.time())))
        if default_param:
            clf = DTC(class_weight='balanced', splitter='best', max_depth=None, max_features=None, max_leaf_nodes=None,
                      min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2,
                      min_weight_fraction_leaf=0.0, presort=False, random_state=seed)
        elif dataset == 'politifact':
            # {'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': None, 'max_features': None,
            #  'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1,
            #  'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'presort': False, 'random_state': 1,
            #  'splitter': 'best'}
            clf = DTC(**{'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': None, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'presort': False, 'random_state': 1, 'splitter': 'best'})
        elif dataset == 'gossipcop':
            # {'class_weight': 'balanced', 'criterion': 'entropy', 'max_depth': None, 'max_features': None,
            #  'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1,
            #  'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'presort': False, 'random_state': 1,
            #  'splitter': 'best'}
            clf = DTC(**{'class_weight': 'balanced', 'criterion': 'entropy', 'max_depth': None, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'presort': False, 'random_state': 1, 'splitter': 'best'})
        elif dataset == 'politifact-gossipcop':
            # {'class_weight': 'balanced', 'criterion': 'entropy', 'max_depth': None, 'max_features': None,
            #  'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1,
            #  'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'presort': False, 'random_state': 1,
            #  'splitter': 'best'}

            clf = DTC(**{'class_weight': 'balanced', 'criterion': 'entropy', 'max_depth': None, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'presort': False, 'random_state': 1, 'splitter': 'best'})
    else:
        print("Start Decision Tree hyperperameter tuning")
        print("Start " + str(datetime.datetime.fromtimestamp(time.time())))

        if cv == 1:
            # tricks to do without cross validation
            clf = GCV(DTC(), parameters, cv=ShuffleSplit(test_size=0.20, n_splits=1), n_jobs=40)
        else:
            clf = GCV(DTC(), parameters, cv=cv, n_jobs=40)
    return clf, "Decision Tree", parameters


def svm(gcv, default_param, dataset, class_weight, seed, cv=1):
    grid_C = [0.1 * i for i in range(1, 21)]
    parameters = {'C': grid_C, 'kernel': ['poly', 'rbf'], 'degree': [ 2, 3],
                  'gamma': ['auto'], 'coef0': [0.01], 'shrinking': [True], 'probability': [False],
                  'tol': [5e-4], 'cache_size': [200],
                  'max_iter': [-1], 'decision_function_shape': ['ovr'], 'random_state': [seed], }
    if class_weight is not False:
        parameters["class_weight"] = [class_weight]
    if not gcv:
        print("Start SVM training")
        print("Start " + str(datetime.datetime.fromtimestamp(time.time())))
        if default_param:
            clf = SVC(random_state=seed,class_weight=class_weight, gamma='auto', coef0=0.01, shrinking=True, tol= 5e-4,
                      probability=False, decision_function_shape= 'ovr', cache_size=200, max_iter=-1)
        elif dataset == 'politifact':
            # {'C': 5.5, 'cache_size': 200, 'class_weight': 'balanced', 'coef0': 0.0, 'decision_function_shape': 'ovr',
            # 'degree': 1, 'gamma': 'auto', 'kernel': 'linear', 'max_iter': 30000, 'probability': False, 'random_state': 1,
            # 'shrinking': True, 'tol': 0.001, 'verbose': False}
            clf = SVC(**{'C': 0.6000000000000001, 'cache_size': 200, 'class_weight': 'balanced', 'coef0': 0.01, 'decision_function_shape': 'ovr', 'degree': 2, 'gamma': 'auto', 'kernel': 'rbf', 'max_iter': -1, 'probability': False, 'random_state': 1, 'shrinking': True, 'tol': 0.0005})
        elif dataset == 'gossipcop':
            # {'C': 9.5, 'cache_size': 200, 'class_weight': 'balanced', 'coef0': 0.0, 'decision_function_shape': 'ovr',
            #  'degree': 1, 'gamma': 'auto', 'kernel': 'linear', 'max_iter': 30000, 'probability': False,
            #  'random_state': 1, 'shrinking': True, 'tol': 0.0005, 'verbose': False}
            clf = SVC(**{'C': 0.30000000000000004, 'cache_size': 200, 'class_weight': 'balanced', 'coef0': 0.01, 'decision_function_shape': 'ovr', 'degree': 2, 'gamma': 'auto', 'kernel': 'poly', 'max_iter': -1, 'probability': False, 'random_state': 1, 'shrinking': True, 'tol': 0.0005})
        elif dataset == 'politifact-gossipcop':
            # {'C': 5.0, 'cache_size': 200, 'class_weight': 'balanced', 'coef0': 0.0, 'decision_function_shape': 'ovr',
            #  'degree': 1, 'gamma': 'auto', 'kernel': 'linear', 'max_iter': 30000, 'probability': False,
            #  'random_state': 1, 'shrinking': True, 'tol': 0.0005, 'verbose': False}
            clf = SVC(**{'C': 2.0, 'cache_size': 200, 'class_weight': 'balanced', 'coef0': 0.01, 'decision_function_shape': 'ovr', 'degree': 2, 'gamma': 'auto', 'kernel': 'rbf', 'max_iter': -1, 'probability': False, 'random_state': 1, 'shrinking': True, 'tol': 0.0005})
    else:
        print("Start SVM hyperperameter tuning")
        print("Start " + str(datetime.datetime.fromtimestamp(time.time())))

        if cv == 1:
            # tricks to do without cross validation
            clf = GCV(SVC(), parameters, cv=ShuffleSplit(test_size=0.20, n_splits=1), n_jobs=40)
        else:
            clf = GCV(SVC(), parameters, cv=cv, n_jobs=40)
    return clf, "SVM", parameters


def random_forest(gcv, default_param, dataset, class_weight, seed, cv=1):
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
                  'oob_score': [False], 'random_state': [seed], 'verbose': [0],
                  'warm_start': [False]}
    if class_weight is not False:
        parameters["class_weight"] = [class_weight]
    if not gcv:
        print("Start Random Forest training")
        print("Start " + str(datetime.datetime.fromtimestamp(time.time())))
        if default_param:
            clf = RandomForestClassifier(class_weight=class_weight, max_depth=None,
                                         max_features='auto', bootstrap=True,
                                         max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                                         min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0,
                                         n_jobs=40, oob_score=False, random_state=seed, verbose=0,
                                         warm_start=False)
        elif dataset == 'politifact':
            # {'bootstrap': True, 'class_weight': 'balanced', 'criterion': 'entropy', 'max_depth': None,
            #  'max_features': 'auto', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None,
            #  'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 50,
            #  'oob_score': False, 'random_state': 1, 'verbose': 0, 'warm_start': False}
            clf = RandomForestClassifier(n_jobs=40, **{'bootstrap': True, 'class_weight': 'balanced', 'criterion': 'entropy', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 50, 'oob_score': False, 'random_state': 1, 'verbose': 0, 'warm_start': False})
        elif dataset == 'gossipcop':
            # {'bootstrap': True, 'class_weight': 'balanced', 'criterion': 'entropy', 'max_depth': None,
            #  'max_features': 'auto', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None,
            #  'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 120,
            #  'oob_score': False, 'random_state': 1, 'verbose': 0, 'warm_start': False}
            clf = RandomForestClassifier(n_jobs=40, **{'bootstrap': True, 'class_weight': 'balanced', 'criterion': 'entropy', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 120, 'oob_score': False, 'random_state': 1, 'verbose': 0, 'warm_start': False})
        elif dataset == "politifact-gossipcop":
            # {'bootstrap': True, 'class_weight': 'balanced', 'criterion': 'entropy', 'max_depth': None,
            #  'max_features': 'auto', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None,
            #  'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 300,
            #  'oob_score': False, 'random_state': 1, 'verbose': 0, 'warm_start': False}
            clf = RandomForestClassifier(n_jobs=40, **{'bootstrap': True, 'class_weight': 'balanced', 'criterion': 'entropy', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 300, 'oob_score': False, 'random_state': 1, 'verbose': 0, 'warm_start': False})
    else:
        print("Start Random Forest hyperperameter tuning")
        print("Start " + str(datetime.datetime.fromtimestamp(time.time())))

        if cv == 1:
            # tricks to do without cross validation
            clf = GCV(RandomForestClassifier(), parameters, cv=ShuffleSplit(test_size=0.20, n_splits=1), n_jobs=40)
        else:
            clf = GCV(RandomForestClassifier(), parameters, cv=cv, n_jobs=40)
    return clf, "Random Forest", parameters


def xgboost(gcv, default_param, dataset, class_weight, seed, cv=1):
    grid_R = [0.1 * i for i in range(1, 10)]
    grid_N = [10 * i for i in range(1, 21)]
    parameters = {'max_depth': [3, 6], 'learning_rate': grid_R, 'n_estimators': grid_N,
                  'objective': ['multi:softmax', 'multi:softprob'], 'booster': ['gbtree', 'dart'],
                  'subsample': [0.7, 0.8, 0.9, 1.], 'random_state': [seed], "num_class": [2, 4, 6]
                  }
    # "scale_pos_weight": [0.31]
    if class_weight is not False:
        parameters["class_weight"] = [class_weight]
    if not gcv:
        print("Start XGBoost training")
        print("Start " + str(datetime.datetime.fromtimestamp(time.time())))
        if default_param:
            clf = xgb.XGBClassifier(random_state=seed, class_weight=class_weight, n_jobs=40)
        elif dataset == 'politifact':
            # {'booster': 'dart', 'class_weight': 'balanced', 'learning_rate': 0.5, 'max_depth': 6, 'n_estimators': 10,
            #  'num_class': 6, 'objective': 'multi:softmax', 'random_state': 1, 'subsample': 0.7}
            clf = xgb.XGBClassifier(n_jobs=40, **{'booster': 'gbtree', 'class_weight': 'balanced', 'learning_rate': 0.7000000000000001, 'max_depth': 3, 'n_estimators': 130, 'num_class': 2, 'objective': 'multi:softmax', 'random_state': 1, 'subsample': 0.8})
        elif dataset == 'gossipcop':
            # {'booster': 'dart', 'class_weight': 'balanced', 'learning_rate': 0.2, 'max_depth': 6, 'n_estimators': 130,
            #  'num_class': 2, 'objective': 'multi:softmax', 'random_state': 1, 'subsample': 0.7}
            clf = xgb.XGBClassifier(n_jobs=40, **{'booster': 'dart', 'class_weight': 'balanced', 'learning_rate': 0.2, 'max_depth': 6, 'n_estimators': 130, 'num_class': 2, 'objective': 'multi:softmax', 'random_state': 1, 'subsample': 0.7})
        elif dataset == 'politifact-gossipcop':
            # {'booster': 'dart', 'class_weight': 'balanced', 'learning_rate': 0.30000000000000004, 'max_depth': 6,
            #  'n_estimators': 120, 'num_class': 6, 'objective': 'multi:softmax', 'random_state': 1, 'subsample': 0.8}
            clf = xgb.XGBClassifier(booster='dart', learning_rate=0.30000000000000004, max_depth=6, n_estimators=120, num_class=6,
                                    objective='multi:softmax', random_state=seed, subsample=0.8, n_jobs=40, verbosity=0,
                                    class_weight=class_weight)

    else:
        print("Start XGBoost hyperperameter tuning")
        print("Start " + str(datetime.datetime.fromtimestamp(time.time())))
        if cv == 1:
            # trick to do this without cross validation
            clf = GCV(xgb.XGBClassifier(), parameters, cv=ShuffleSplit(test_size=0.20, n_splits=1), n_jobs=40)
        else:
            clf = GCV(xgb.XGBClassifier(), parameters, cv=cv, n_jobs=40)
    return clf, "XGboost", parameters


def logistic_reg(gcv, default_param,dataset, class_weight, seed, cv=1):
    grid_C = [0.5 * i for i in range(1, 21)]
    parameters = {"tol": [5e-4], "C": grid_C, "random_state": [seed],
                  "solver": ["newton-cg", "sag", "saga", "lbfgs"],
                  "max_iter": [4000], "multi_class": ["multinomial", "ovr", "auto"]}
    if class_weight is not False:
        parameters["class_weight"] = [class_weight]
    if not gcv:
        print("Start Logistic Regression training")
        print("Start " + str(datetime.datetime.fromtimestamp(time.time())))
        if default_param:
            clf = LogisticRegression(tol=5e-4, max_iter=4000, random_state=seed, class_weight=class_weight, n_jobs=40)
        elif dataset == 'politifact':
            # {'C': 7.0, 'class_weight': 'balanced', 'max_iter': 4000, 'multi_class': 'multinomial', 'random_state': 1, 'solver': 'newton-cg', 'tol': 0.0005}
            clf = LogisticRegression(n_jobs=40, **{'C': 7.0, 'class_weight': 'balanced', 'max_iter': 4000, 'multi_class': 'multinomial', 'random_state': 1, 'solver': 'newton-cg', 'tol': 0.0005})
        elif dataset == 'gossipcop':
            # {'C': 10.0, 'class_weight': 'balanced', 'max_iter': 4000, 'multi_class': 'multinomial', 'random_state': 1,
            #  'solver': 'sag', 'tol': 0.0005}
            clf = LogisticRegression(n_jobs=40, **{'C': 10.0, 'class_weight': 'balanced', 'max_iter': 4000, 'multi_class': 'multinomial', 'random_state': 1, 'solver': 'sag', 'tol': 0.0005})
        elif dataset == 'politifact-gossipcop':
            # {'C': 7.0, 'class_weight': 'balanced', 'max_iter': 4000, 'multi_class': 'multinomial', 'random_state': 1,
            #  'solver': 'sag', 'tol': 0.0005}
            clf = LogisticRegression(n_jobs=40, **{'C': 7.0, 'class_weight': 'balanced', 'max_iter': 4000, 'multi_class': 'multinomial', 'random_state': 1, 'solver': 'sag', 'tol': 0.0005})

    else:
        print("Start Logistic Regression hyperperameter tuning")
        print("Start " + str(datetime.datetime.fromtimestamp(time.time())))
        if cv == 1:
            # trick to do this without cross validation
            clf = GCV(LogisticRegression(), parameters, cv=ShuffleSplit(test_size=0.20, n_splits=1), n_jobs=40)
        else:
            clf = GCV(LogisticRegression(), parameters, cv=cv, n_jobs=40)
    return clf, "Logistic Regression", parameters
