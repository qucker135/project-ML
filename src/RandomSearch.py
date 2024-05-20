import copy
import random
from sklearn.model_selection import KFold

def sample_from_range(param_range):
    if isinstance(param_range, list) and len(param_range) == 2:
        lower, upper = param_range
        if all(isinstance(bound, int) for bound in param_range):
            return random.randint(lower, upper)
        else:
            return random.uniform(lower, upper)
    else:
        raise ValueError("Invalid parameter range format. Expected a list with two elements.")

def RandomSearchCustom(df, num_splits, estimator, param_ranges, scoring, target_column, verbose, n_iter):
    best_score = -float('inf')
    best_params = None

    X = df.drop(columns=[target_column])
    y = df[target_column]

    kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

    for i in range(n_iter):
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
            print(f"Iteration {i+1}/{n_iter}, Best Score: {best_score}, Best Params: {best_params}, Params: {params}, avg score: {avg_score}")

    return best_params, best_score