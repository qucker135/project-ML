import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from GridSearch import GridSearchCustom
from RandomSearch import RandomSearchCustom
from RandomGridCombined import RandomSearchWithGridSearch
from RandomSearchModified import RandomSearchModified
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import matthews_corrcoef, make_scorer

df = pd.read_csv('datasets/nasa.csv')

estimator = RandomForestClassifier()

param_grid = {
    'n_estimators': [10],
    'max_depth': [9, 10, 11],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 3, 6],
    'max_features': [2, 5, 10, 20],
    'max_leaf_nodes': [500, 750, 1000],
    'min_impurity_decrease': [0, 0.001]
}

param_ranges = {
    'n_estimators': [10, 10],
    'max_depth': [9, 11],
    'min_samples_split': [2, 20],
    'min_samples_leaf': [1, 6],
    'max_features': [2, 20],
    'max_leaf_nodes': [500, 1000],
    'min_impurity_decrease': [0, 0.001]
}


num_splits = 5

scoring = matthews_corrcoef

verbose = True

if __name__ == '__main__':
    # best_params, best_score = GridSearchCustom(df, num_splits, estimator, param_grid, scoring, 'Hazardous', verbose)

    # best_params, best_score = RandomSearchCustom(df, num_splits, estimator, param_ranges, scoring, 'Hazardous', verbose, 15)

    best_params, best_score = RandomSearchWithGridSearch(df, num_splits, estimator, param_ranges, scoring, 'Hazardous', verbose, 3, 25)

    # best_params, best_score = RandomSearchModified(df, num_splits, estimator, param_ranges, scoring, 'Hazardous', verbose, 10, 15)

    print("Best Parameters:", best_params)
    print("Best Score:", best_score)