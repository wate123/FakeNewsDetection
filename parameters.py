from sklearn.linear_model import LogisticRegression


#tuning hyperparameters
grid_R = [0.1 * i for i in range(1, 10)]
grid_N = [10 * i for i in range(1, 21)]

'''
Logistic Regression
'''
# clf = LogisticRegression(tol=0.0005, C=0.5, max_iter=4000, multi_class='ovr', random_state=1, solver='saga')
# clf = GCV(LogisticRegression(), parameters, cv=10, n_jobs=40)

# grid_C = [0.5 * i for i in range(1, 21)]
# parameters = {"tol": [5e-4], "C": grid_C, "random_state": [1],
#               "solver": ["newton-cg", "sag", "saga", "lbfgs"],
#               "max_iter": [4000], "multi_class": ["multinomial", "ovr", "auto"]}

'''
Gradient Boosting
'''
# parameters = {'loss': 'deviance', 'max_depth': 80, 'max_features': 'auto', 'min_samples_leaf': 4, 'min_samples_split': 4, 'n_estimators': 1000, 'random_state': 42}
# clf = GradientBoostingClassifier()
# clf = GCV(GradientBoostingClassifier(), parameters, cv=2, n_jobs=40)

'''
XGBoost
'''
# xgboost tuning
parameters = {'max_depth': [3,6], 'learning_rate': grid_R, 'n_estimators': grid_N,
              'objective': ['multi:softmax', 'multi:softprob'], 'booster': ['gbtree', 'dart'],
              'subsample': [0.7, 0.8, 0.9, 1.], 'random_state': [1], "num_class": [2, 4, 6]}


'''
Random Forest
'''
# n_estimators = [500]
# max_features = ['auto']
# # # # Maximum number of levels in tree
# max_depth = [10]
# # # #max_depth.append(None)
# # # # Minimum number of samples required to split a node
# min_samples_split = [2]
# # # # Minimum number of samples required at each leaf node
# min_samples_leaf = [2]
# # # # Method of selecting samples for training each tree
# bootstrap = [False]
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'random_state':[1],
#                'bootstrap': bootstrap}
#
#
#
# rf = RF()
# rf_random = GCV(estimator = rf, param_grid=random_grid, verbose=2, cv=10, n_jobs=40 )
