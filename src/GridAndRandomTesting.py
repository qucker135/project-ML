import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from GridSearch import GridSearchCustom
from RandomSearch import RandomSearchCustom
from RandomGridCombined import RandomSearchWithGridSearch
from RandomSearchModified import RandomSearchModified

df = pd.read_csv('../datasets/nasa_cleaned.csv')

estimator = RandomForestClassifier()

param_grid = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [None, 10, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [0.1, 0.5, 1.0]
}

param_ranges = {
    'n_estimators': [10, 500],
    'max_depth': [5, 30],
    'min_samples_split': [2, 11],
    'min_samples_leaf': [1, 5],
    'max_features': [0.1, 1.0]
}


num_splits = 5

scoring = accuracy_score

verbose = True

if __name__ == '__main__':

    #best_params, best_score = GridSearchCustom(df, num_splits, estimator, param_grid, scoring, 'Hazardous', verbose)

    best_params, best_score = RandomSearchCustom(df, num_splits, estimator, param_ranges, scoring, 'Hazardous', verbose, 25)

    #best_params, best_score = RandomSearchWithGridSearch(df, num_splits, estimator, param_ranges, scoring, 'Hazardous', verbose, 10, 25)

    #best_params, best_score = RandomSearchModified(df, num_splits, estimator, param_ranges, scoring, 'Hazardous', verbose, 10, 15)

    print("Best Parameters:", best_params)
    print("Best Score:", best_score)