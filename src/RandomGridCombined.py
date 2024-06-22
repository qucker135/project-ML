import numpy as np
import itertools

from matplotlib import pyplot as plt
from numpy import interp
from sklearn.metrics import matthews_corrcoef, auc, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold
from gridsearch_randomsearch_helpers import dropped_columns, unnecessary_columns

from RandomSearch import sample_from_range

def RandomSearchWithGridSearch(df, num_splits, estimator, param_ranges, scoring, target_column, verbose, n_iter_random, step_size=0.1, num_grid_points=3):
    df_X = df.drop(columns=[target_column])
    df_y = df[target_column]

    kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fold_counter = 0
    plt.figure(figsize=(10, 10), dpi=400)

    predictions_nemar = []
    predictions_y_test = []

    for (train_index, test_index) in kf.split(df_X, df_y):
        best_score = None
        best_params = None

        df_X_train = df_X.iloc[train_index]
        df_y_train = df_y.iloc[train_index]
        df_X_test = df_X.iloc[test_index]
        df_y_test = df_y.iloc[test_index]

        df_X_train.drop(columns=(dropped_columns+unnecessary_columns), inplace=True)
        df_X_test.drop(columns=(dropped_columns+unnecessary_columns), inplace=True)

        rskf = RepeatedStratifiedKFold(n_splits=num_splits, random_state=42)

        for i in range(n_iter_random):

            params = {}
            for param_name, param_range in param_ranges.items():
                params[param_name] = sample_from_range(param_range)

            estimator.set_params(**params)

            fold_scores = []

            for train_idx, val_idx in rskf.split(df_X_train, df_y_train):
                X_train, X_val = df_X_train.iloc[train_idx], df_X_train.iloc[val_idx]
                y_train, y_val = df_y_train.iloc[train_idx], df_y_train.iloc[val_idx]

                estimator.fit(X_train, y_train)

                y_pred = estimator.predict(X_val)

                score = scoring(y_val, y_pred)
                fold_scores.append(score)

            avg_score = sum(fold_scores) / len(fold_scores)

            if best_score is None or avg_score > best_score:
                best_score = avg_score
                best_params = params

            if verbose:
                print(f"Iteration {i + 1}/{n_iter_random}, Best Score: {best_score}, Best Params: {best_params}, Params: {params}, avg score: {avg_score}")

        param_grid = {}
        for param_name, param_range in param_ranges.items():
            if isinstance(param_range, list) and len(param_range) == 2:
                lower_bound, upper_bound = param_range
                if isinstance(lower_bound, int) and isinstance(upper_bound, int):
                    best_value = best_params[param_name]
                    start_value = max(int(best_value * (1 - step_size)), lower_bound)
                    end_value = min(int(best_value * (1 + step_size)), upper_bound)
                    param_values = np.linspace(start_value, end_value, num_grid_points, dtype=int)
                    param_grid[param_name] = list(set(param_values.tolist()))  # Convert to set to remove duplicates
                else:
                    best_value = best_params[param_name]
                    start_value = max(best_value * (1 - step_size), lower_bound)
                    end_value = min(best_value * (1 + step_size), upper_bound)
                    param_values = np.linspace(start_value, end_value, num_grid_points)
                    param_grid[param_name] = list(set(param_values.tolist()))  # Convert to set to remove duplicates

        combinations_list = list(itertools.product(*param_grid.values()))

        print(param_grid)

        best_score = None

        for combination in combinations_list:
            estimator.set_params(**dict(zip(param_grid.keys(), combination)))

            fold_scores = []

            for train_idx, val_idx in rskf.split(df_X_train, df_y_train):
                X_train, X_val = df_X_train.iloc[train_idx], df_X_train.iloc[val_idx]
                y_train, y_val = df_y_train.iloc[train_idx], df_y_train.iloc[val_idx]

                estimator.fit(X_train, y_train)
                y_pred = estimator.predict(X_val)
                score = scoring(y_val, y_pred)
                fold_scores.append(score)

            avg_score = sum(fold_scores) / len(fold_scores)

            if verbose:
                print("Combination:", combination)
                print("Score:", avg_score)

            if best_score is None or avg_score > best_score:
                best_score = avg_score
                best_params = combination

        estimator.set_params(**dict(zip(param_grid.keys(), best_params)))
        estimator.fit(df_X_train, df_y_train)

        y_prob = estimator.predict_proba(df_X_test)[:, 1]

        print(estimator.score(df_X_test, df_y_test))
        print(matthews_corrcoef(df_y_test, estimator.predict(df_X_test)))
        print((estimator.predict(df_X_test) == df_y_test).value_counts())

        predictions_nemar.append(estimator.predict(df_X_test))
        predictions_y_test.append(df_y_test)

        fpr, tpr, thresholds = roc_curve(df_y_test, y_prob)

        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        print(fold_counter, tpr, fpr)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (fold_counter, roc_auc))

        fold_counter += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title('Cross-Validation ROC of RandomSearch', fontsize=18)
    plt.legend(loc="lower right", prop={'size': 5})
    plt.savefig(fname='AUC-ROC_RS')
    plt.show()

    predictions_nemar_con = np.concatenate(predictions_nemar)
    predictions_y_con = np.concatenate(predictions_y_test)

    return best_params, best_score, predictions_nemar_con, predictions_y_con
