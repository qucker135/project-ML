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

# param_grid = {
#     'n_estimators': [10],
#     'max_depth': [9, 10, 11],
#     'min_samples_split': [2, 5, 10, 20],
#     'min_samples_leaf': [1, 3, 6],
#     'max_features': [2, 5, 10, 20],
#     'max_leaf_nodes': [500, 750, 1000],
#     'min_impurity_decrease': [0, 0.001]
# }

param_grid = {
    'n_estimators': [10],
    'max_depth': [9, 11],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 3, 6],
    'max_features': [2, 10, 20],
    'max_leaf_nodes': [500, 750],
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


num_splits = 10

scoring = matthews_corrcoef

verbose = True

if __name__ == '__main__':
    # best_params, best_score, predictions_nemar_con, predictions_y_con = GridSearchCustom(df, num_splits, estimator, param_grid, scoring, 'Hazardous', verbose)

    # best_params, best_score, predictions_nemar_con, predictions_y_con = RandomSearchCustom(df, num_splits, estimator, param_ranges, scoring, 'Hazardous', verbose, n_iter=25)

    best_params, best_score, predictions_nemar_con, predictions_y_con = RandomSearchWithGridSearch(df, num_splits, estimator, param_ranges, scoring, 'Hazardous', verbose, num_grid_points=3, n_iter_random=25)

    # best_params, best_score, predictions_nemar_con, predictions_y_con = RandomSearchModified(df, num_splits, estimator, param_ranges, scoring, 'Hazardous', verbose, n_iter_inital=10, n_iter_refined=15)

    print("Best Parameters:", best_params)
    print("Best Score:", best_score)