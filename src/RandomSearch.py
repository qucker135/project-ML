import random

from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import RepeatedStratifiedKFold

from gridsearch_randomsearch_helpers import dropped_columns, unnecessary_columns

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
    df_X = df.drop(columns=[target_column])
    df_y = df[target_column]

    kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

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

        for i in range(n_iter):

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
                print(
                    f"Iteration {i + 1}/{n_iter}, Best Score: {best_score}, Best Params: {best_params}, Params: {params}, avg score: {avg_score}")

        estimator.set_params(**params)
        estimator.fit(df_X_train, df_y_train)
        print(estimator.score(df_X_test, df_y_test))
        print(matthews_corrcoef(df_y_test, estimator.predict(df_X_test)))
        print((estimator.predict(df_X_test) == df_y_test).value_counts())

        return best_params, best_score



