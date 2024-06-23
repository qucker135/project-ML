import itertools
from main2 import main
import pandas as pd
import numpy as np
from settings import mcnemar_test, get_contingency_table
from GridSearch import GridSearchCustom
from RandomSearch import RandomSearchCustom
from RandomGridCombined import RandomSearchWithGridSearch
from RandomSearchModified import RandomSearchModified
from GridAndRandomTesting2 import df, num_splits, estimator, param_grid, scoring, verbose, param_ranges

# ROOT_DIR = Path(__file__).resolve().parent.parent
# DATASET_PATH_NASA = ROOT_DIR / 'datasets' / 'nasa.csv'
# # https://www.kaggle.com/datasets/ritwikb3/heart-disease-statlog?resource=download
# DATASET_PATH_HEART = ROOT_DIR / 'datasets' / 'Heart_disease_statlog.csv'
# # https://www.kaggle.com/datasets/matinmahmoudi/sales-and-satisfaction?select=Sales_without_NaNs_v1.3.csv
# DATASET_PATH_SALES = ROOT_DIR / 'datasets' / 'Sales_without_NaNs_v1.3.csv'

# algs = ['genetic', 'random', 'random_mod', 'random_grid', 'bayes', 'grid']

algs = ['random', 'random_mod']

if __name__ == "__main__":
    p_values = {}

    nemar_predictions = []
    y_predictions = []

    for model in algs:
        if model == 'bayes' or model == 'genetic':
            predictions_nemar_con, predictions_y_con = main(model)
        elif model == 'grid':
            best_params, best_score, predictions_nemar_con, predictions_y_con = GridSearchCustom(df, num_splits, estimator, param_grid, scoring, 'Hazardous', verbose)
        elif model == 'random':
            best_params, best_score, predictions_nemar_con, predictions_y_con = RandomSearchCustom(df, num_splits, estimator, param_ranges, scoring, 'Hazardous', verbose, n_iter=2) # !!!!!!!!!!!!!!
        elif model == 'random_mod':
            best_params, best_score, predictions_nemar_con, predictions_y_con = RandomSearchModified(df, num_splits, estimator, param_ranges, scoring, 'Hazardous', verbose, n_iter_initial=10, n_iter_refined=15)
        elif model == 'random_grid':
            best_params, best_score, predictions_nemar_con, predictions_y_con = RandomSearchWithGridSearch(df, num_splits, estimator, param_ranges, scoring, 'Hazardous', verbose, num_grid_points=3, n_iter_random=25)

        nemar_predictions.append(predictions_nemar_con)
        y_predictions.append(predictions_y_con)

    for i, j in itertools.combinations_with_replacement(range(len(algs)), 2):
        Y_ = pd.DataFrame(y_predictions[i], columns=['Hazardous'])
        Y_[algs[i]] = np.where(nemar_predictions[i] > 0.5, 1, 0)
        Y_[algs[j]] = np.where(nemar_predictions[j] > 0.5, 1, 0)

        p_values[(algs[i], algs[j])] = mcnemar_test(get_contingency_table(Y_, 'Hazardous', algs[i], algs[j]))
    
    # for model1, model2 in itertools.combinations_with_replacement(algs, 2):
    #     if model1 == 'bayes' or model1 == 'genetic':
    #         predictions_nemar_con_model1, predictions_y_con_model1 = main(model1)
    #     elif model1 == 'grid':
    #         best_params_model1, best_score_model1, predictions_nemar_con_model1, predictions_y_con_model1 = GridSearchCustom(df, num_splits, estimator, param_grid, scoring, 'Hazardous', verbose)
    #     elif model1 == 'random':
    #         best_params_model1, best_score_model1, predictions_nemar_con_model1, predictions_y_con_model1 = RandomSearchCustom(df, num_splits, estimator, param_ranges, scoring, 'Hazardous', verbose, 15)
    #     elif model1 == 'random_mod':
    #         best_params_model1, best_score_model1, predictions_nemar_con_model1, predictions_y_con_model1 = RandomSearchModified(df, num_splits, estimator, param_ranges, scoring, 'Hazardous', verbose, 10, 15)
    #     elif model1 == 'random_grid':
    #         best_params_model1, best_score_model1, predictions_nemar_con_model1, predictions_y_con_model1 = RandomSearchWithGridSearch(df, num_splits, estimator, param_ranges, scoring, 'Hazardous', verbose, 3, 25)
# 
    #     # the same for model2
    #     if model2 == 'bayes' or model2 == 'genetic':
    #         predictions_nemar_con_model2, predictions_y_con_model2 = main(model2)
    #     elif model2 == 'grid':
    #         best_params_model2, best_score_model2, predictions_nemar_con_model2, predictions_y_con_model2 = GridSearchCustom(df, num_splits, estimator, param_grid, scoring, 'Hazardous', verbose)
    #     elif model2 == 'random':
    #         best_params_model2, best_score_model2, predictions_nemar_con_model2, predictions_y_con_model2 = RandomSearchCustom(df, num_splits, estimator, param_ranges, scoring, 'Hazardous', verbose, 15)
    #     elif model2 == 'random_mod':
    #         best_params_model2, best_score_model2, predictions_nemar_con_model2, predictions_y_con_model2 = RandomSearchModified(df, num_splits, estimator, param_ranges, scoring, 'Hazardous', verbose, 10, 15)
    #     elif model2 == 'random_grid':
    #         best_params_model2, best_score_model2, predictions_nemar_con_model2, predictions_y_con_model2 = RandomSearchWithGridSearch(df, num_splits, estimator, param_ranges, scoring, 'Hazardous', verbose, 3, 25)
    #     
    #     Y_ = pd.DataFrame(predictions_y_con_model1, columns=['Hazardous'])
    #     Y_[model1] = np.where(predictions_nemar_con_model1 > 0.5, 1, 0)
    #     Y_[model2] = np.where(predictions_nemar_con_model2 > 0.5, 1, 0)
# 
    #     p_values[(model1, model2)] = mcnemar_test(get_contingency_table(Y_, 'Hazardous', model1, model2))

    print(p_values)
    