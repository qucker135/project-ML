import itertools
from sklearn.model_selection import RepeatedStratifiedKFold
from gridsearch_randomsearch_helpers import dropped_columns, unnecessary_columns
from sklearn.metrics import matthews_corrcoef

def GridSearchCustom(df, num_splits, estimator, param_grid, scoring, target_column, verbose):
    param_values = param_grid.values()
    combinations_list = list(itertools.product(*param_values))

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

        for combination in combinations_list:
            fold_scores = []
            for train_idx, val_idx in rskf.split(df_X_train, df_y_train):
                X_train, X_val = df_X_train.iloc[train_idx], df_X_train.iloc[val_idx]
                y_train, y_val = df_y_train.iloc[train_idx], df_y_train.iloc[val_idx]

                estimator.set_params(**dict(zip(param_grid.keys(), combination)))

                estimator.fit(X_train, y_train)

                y_pred = estimator.predict(X_val)

                score = scoring(y_val, y_pred)
                fold_scores.append(score)

            avg_score = sum(fold_scores) / len(fold_scores)

            if verbose:
                print("Combination:", combination)
                print("Average Score:", avg_score)

            if best_score is None or avg_score > best_score:
                best_score = avg_score
                best_params = combination

        estimator.set_params(**dict(zip(param_grid.keys(), best_params)))
        estimator.fit(df_X_train, df_y_train)
        print(estimator.score(df_X_test, df_y_test))
        print(matthews_corrcoef(df_y_test, estimator.predict(df_X_test)))
        print((estimator.predict(df_X_test) == df_y_test).value_counts())

        # score

    return best_params, best_score