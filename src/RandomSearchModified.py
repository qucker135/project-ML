import copy
import random
import numpy as np
from sklearn.model_selection import KFold
from RandomSearch import sample_from_range

def RandomSearchModified(df, num_splits, estimator, param_ranges, scoring, target_column, verbose, n_iter_initial, n_iter_refined, top_percentage=0.1, refine_range_percentage=0.2):
    best_score = -float('inf')
    best_params = None

    X = df.drop(columns=[target_column])
    y = df[target_column]

    kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

    top_results = []

    # Initial random search
    for i in range(n_iter_initial):
        params = {}
        for param_name, param_range in param_ranges.items():
            params[param_name] = sample_from_range(param_range)

        estimator.set_params(**params)

        fold_scores = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            estimator.fit(X_train, y_train)
            y_pred = estimator.predict(X_val)
            score = scoring(y_val, y_pred)
            fold_scores.append(score)

        avg_score = sum(fold_scores) / num_splits
        top_results.append((avg_score, copy.deepcopy(params)))

        if avg_score > best_score:
            best_score = avg_score
            best_params = copy.deepcopy(params)

        if verbose:
            print(f"Iteration {i+1}/{n_iter_initial}, Best Score: {best_score}, Best Params: {best_params}, Params: {params}, avg score: {avg_score}")

    top_results.sort(key=lambda x: x[0], reverse=True)
    num_top_results = 3
    top_results = top_results[:num_top_results]

    # Refine search based on top results
    for top_result in top_results:
        for i in range(n_iter_refined):
            params = {}
            for param_name, param_range in param_ranges.items():
                if isinstance(param_range, list) and len(param_range) == 2:
                    lower_bound, upper_bound = param_range
                    typez = type(lower_bound)
                    base_value = top_result[1][param_name]
                    lower_bound = max(lower_bound, base_value - refine_range_percentage * (upper_bound - lower_bound))
                    upper_bound = min(upper_bound, base_value + refine_range_percentage * (upper_bound - lower_bound))
                    param_range = [lower_bound, upper_bound]
                    if typez == int:
                        param_range = [int(lower_bound), int(upper_bound)]
                params[param_name] = sample_from_range(param_range)

            estimator.set_params(**params)

            fold_scores = []
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                estimator.fit(X_train, y_train)
                y_pred = estimator.predict(X_val)
                score = scoring(y_val, y_pred)
                fold_scores.append(score)

            avg_score = sum(fold_scores) / num_splits

            if avg_score > best_score:
                best_score = avg_score
                best_params = copy.deepcopy(params)

            if verbose:
                print(f"Iteration {i+1}/{n_iter_refined}, Best Score: {best_score}, Best Params: {best_params}, Params: {params}, avg score: {avg_score}")

    return best_params, best_score