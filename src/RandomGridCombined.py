import copy
import numpy as np
import itertools
from sklearn.model_selection import KFold

from RandomSearch import sample_from_range

def RandomSearchWithGridSearch(df, num_splits, estimator, param_ranges, scoring, target_column, verbose, n_iter_random, n_iter_grid, step_size=0.1, num_grid_points=3):
    best_score = -float('inf')
    best_params = None

    X = df.drop(columns=[target_column])
    y = df[target_column]

    kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

    # Random Search
    for i in range(n_iter_random):
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

        if avg_score > best_score:
            best_score = avg_score
            best_params = copy.deepcopy(params)

        if verbose:
            print(f"Iteration {i + 1}/{n_iter_random}, Best Score: {best_score}, Best Params: {best_params}, Params: {params}, avg score: {avg_score}")

    print(f"Best parameters from random search: {best_params}, Best score: {best_score}")

    # Grid Search
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

    for combination in combinations_list:
        fold_scores = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            estimator.set_params(**dict(zip(param_grid.keys(), combination)))

            estimator.fit(X_train, y_train)

            y_pred = estimator.predict(X_val)

            score = scoring(y_val, y_pred)
            fold_scores.append(score)

        avg_score = sum(fold_scores) / num_splits

        if verbose:
            print("Combination:", combination)
            print("Score:", avg_score)

        if best_score is None or avg_score > best_score:
            best_score = avg_score
            best_params = combination

    return best_params, best_score

#Memetic optimization