import copy

from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import RepeatedStratifiedKFold
from RandomSearch import sample_from_range
from gridsearch_randomsearch_helpers import dropped_columns, unnecessary_columns

def RandomSearchModified(df, num_splits, estimator, param_ranges, scoring, target_column, verbose, n_iter_initial, n_iter_refined, refine_range_percentage=0.2):
    df_X = df.drop(columns=[target_column])
    df_y = df[target_column]

    kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    for (train_index, test_index) in kf.split(df_X, df_y):
        top_results = []
        best_score = None
        best_params = None

        df_X_train = df_X.iloc[train_index]
        df_y_train = df_y.iloc[train_index]
        df_X_test = df_X.iloc[test_index]
        df_y_test = df_y.iloc[test_index]

        df_X_train.drop(columns=(dropped_columns+unnecessary_columns), inplace=True)
        df_X_test.drop(columns=(dropped_columns+unnecessary_columns), inplace=True)

        rskf = RepeatedStratifiedKFold(n_splits=num_splits, random_state=42)

        for i in range(n_iter_initial):
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
            top_results.append((avg_score, copy.deepcopy(params)))

            if best_score is None or avg_score > best_score:
                best_score = avg_score
                best_params = copy.deepcopy(params)

            if verbose:
                print(f"Iteration {i + 1}/{n_iter_initial}, Best Score: {best_score}, Best Params: {best_params}, Params: {params}, avg score: {avg_score}")

            top_results.sort(key=lambda x: x[0], reverse=True)
            num_top_results = 3
            top_results = top_results[:num_top_results]

            for top_result in top_results:
                for i in range(n_iter_refined):
                    params = {}
                    for param_name, param_range in param_ranges.items():
                        if isinstance(param_range, list) and len(param_range) == 2:
                            lower_bound, upper_bound = param_range
                            typez = type(lower_bound)
                            base_value = top_result[1][param_name]
                            lower_bound = max(lower_bound,
                                              base_value - refine_range_percentage * (upper_bound - lower_bound))
                            upper_bound = min(upper_bound,
                                              base_value + refine_range_percentage * (upper_bound - lower_bound))
                            param_range = [lower_bound, upper_bound]
                            if typez == int:
                                param_range = [int(lower_bound), int(upper_bound)]
                        params[param_name] = sample_from_range(param_range)

                    estimator.set_params(**params)

                    for train_idx, val_idx in rskf.split(df_X_train, df_y_train):
                        X_train, X_val = df_X_train.iloc[train_idx], df_X_train.iloc[val_idx]
                        y_train, y_val = df_y_train.iloc[train_idx], df_y_train.iloc[val_idx]

                        estimator.fit(X_train, y_train)

                        y_pred = estimator.predict(X_val)

                        score = scoring(y_val, y_pred)
                        fold_scores.append(score)

                    avg_score = sum(fold_scores) / len(fold_scores)
                    top_results.append((avg_score, copy.deepcopy(params)))

                    if best_score is None or avg_score > best_score:
                        best_score = avg_score
                        best_params = copy.deepcopy(params)

                    if verbose:
                        print(
                            f"Iteration {i + 1}/{n_iter_refined}, Best Score: {best_score}, Best Params: {best_params}, Params: {params}, avg score: {avg_score}")

        estimator.set_params(**best_params)
        estimator.fit(df_X_train, df_y_train)
        print(estimator.score(df_X_test, df_y_test))
        print(matthews_corrcoef(df_y_test, estimator.predict(df_X_test)))
        print((estimator.predict(df_X_test) == df_y_test).value_counts())

    return best_params, best_score






    # # Initial random search
    # for i in range(n_iter_initial):
    #     params = {}
    #     for param_name, param_range in param_ranges.items():
    #         params[param_name] = sample_from_range(param_range)
    #
    #     estimator.set_params(**params)
    #
    #     fold_scores = []
    #     for train_idx, val_idx in kf.split(X_train_split):
    #         X_train, X_val = X_train_split.iloc[train_idx], X_train_split.iloc[val_idx]
    #         y_train, y_val = y_train_split.iloc[train_idx], y_train_split.iloc[val_idx]
    #
    #         estimator.fit(X_train, y_train)
    #         y_pred = estimator.predict(X_val)
    #         score = scoring._score_func(y_val, y_pred)
    #         fold_scores.append(score)
    #
    #     avg_score = sum(fold_scores) / num_splits
    #     top_results.append((avg_score, copy.deepcopy(params)))
    #
    #     if avg_score > best_score:
    #         best_score = avg_score
    #         best_params = copy.deepcopy(params)
    #
    #     if verbose:
    #         print(f"Iteration {i+1}/{n_iter_initial}, Best Score: {best_score}, Best Params: {best_params}, Params: {params}, avg score: {avg_score}")
    #
    # top_results.sort(key=lambda x: x[0], reverse=True)
    # num_top_results = 3
    # top_results = top_results[:num_top_results]
    #
    # # Refine search based on top results
    # for top_result in top_results:
    #     for i in range(n_iter_refined):
    #         params = {}
    #         for param_name, param_range in param_ranges.items():
    #             if isinstance(param_range, list) and len(param_range) == 2:
    #                 lower_bound, upper_bound = param_range
    #                 typez = type(lower_bound)
    #                 base_value = top_result[1][param_name]
    #                 lower_bound = max(lower_bound, base_value - refine_range_percentage * (upper_bound - lower_bound))
    #                 upper_bound = min(upper_bound, base_value + refine_range_percentage * (upper_bound - lower_bound))
    #                 param_range = [lower_bound, upper_bound]
    #                 if typez == int:
    #                     param_range = [int(lower_bound), int(upper_bound)]
    #             params[param_name] = sample_from_range(param_range)
    #
    #         estimator.set_params(**params)
    #
    #         fold_scores = []
    #         for train_idx, val_idx in kf.split(X_train_split):
    #             X_train, X_val = X_train_split.iloc[train_idx], X_train_split.iloc[val_idx]
    #             y_train, y_val = y_train_split.iloc[train_idx], y_train_split.iloc[val_idx]
    #
    #             estimator.fit(X_train, y_train)
    #             y_pred = estimator.predict(X_val)
    #             score = scoring._score_func(y_val, y_pred)
    #             fold_scores.append(score)
    #
    #         avg_score = sum(fold_scores) / num_splits
    #
    #         if avg_score > best_score:
    #             best_score = avg_score
    #             best_params = copy.deepcopy(params)
    #             best_model = estimator
    #
    #         if verbose:
    #             print(f"Iteration {i+1}/{n_iter_refined}, Best Score: {best_score}, Best Params: {best_params}, Params: {params}, avg score: {avg_score}")
    #
    # return best_params, best_score, best_model