from pathlib import Path
import pandas as pd
import numpy as np
import datetime
import scipy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from BayesianOpt import BayesOpt
from Genetic import Genetic
from settings import domain

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH_NASA = ROOT_DIR / 'datasets' / 'nasa.csv'
# https://www.kaggle.com/datasets/ritwikb3/heart-disease-statlog?resource=download
DATASET_PATH_HEART = ROOT_DIR / 'datasets' / 'Heart_disease_statlog.csv'
# https://www.kaggle.com/datasets/matinmahmoudi/sales-and-satisfaction?select=Sales_without_NaNs_v1.3.csv
DATASET_PATH_SALES = ROOT_DIR / 'datasets' / 'Sales_without_NaNs_v1.3.csv'

# Dropped because they are identifiers, duplicate data or single value columns:
dropped_columns_nasa = [
    'Neo Reference ID',
    'Name',
    'Est Dia in KM(min)',
    'Est Dia in KM(max)',
    'Est Dia in Miles(min)',
    'Est Dia in Miles(max)',
    'Est Dia in Feet(min)',
    'Est Dia in Feet(max)',
    'Close Approach Date',
    'Relative Velocity km per hr',
    'Miles per hour',
    'Miss Dist.(miles)',
    'Miss Dist.(kilometers)',
    'Miss Dist.(Astronomical)',
    'Orbiting Body',
    'Orbit ID',
    'Equinox',
]

# Potentially unnecessary columns:
unnecessary_columns_nasa = [
    'Absolute Magnitude',
    'Orbit Determination Date',
]

if __name__ == "__main__":
    query_dataset = input("Which dataset do you want to use? (nasa | heart | sales):")
    if query_dataset == 'nasa':
        df = pd.read_csv(DATASET_PATH_NASA)
        df_X = df.drop('Hazardous', axis=1)
        df_y = df['Hazardous']
    elif query_dataset == 'heart':
        df = pd.read_csv(DATASET_PATH_HEART)
        df_X = df.drop('target', axis=1)
        df_y = df['target']
    elif query_dataset == 'sales':
        df = pd.read_csv(DATASET_PATH_SALES)
        df.replace({'Yes': 1, 'No': 0, 'Control': 0, 'Treatment': 1, 'High Value': 2, 'Medium Value': 1, 'Low Value': 0}, inplace=True)
        print(df.head())
        df_X = df.drop('Purchase_Made', axis=1)
        df_y = df['Purchase_Made']
    else:
        raise ValueError("Invalid dataset")

    # df = pd.read_csv(DATASET_PATH_NASA)
    # split X, y
    

    query = input("What do you want to do? (bayes/genetic):")

    # RepeatedStratifiedKFold here
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    for (train_index, test_index) in rskf.split(df_X, df_y):
        df_X_train = df_X.iloc[train_index]
        df_y_train = df_y.iloc[train_index]
        df_X_test = df_X.iloc[test_index]
        df_y_test = df_y.iloc[test_index]

        if query_dataset == 'nasa':
            df_X_train.drop(columns=(dropped_columns_nasa+unnecessary_columns_nasa), inplace=True)
            df_X_test.drop(columns=(dropped_columns_nasa+unnecessary_columns_nasa), inplace=True)

        if 'bayes' in query:
            bayes_opt1 = BayesOpt(
                init_guesses=3,
                n_iter=10,
                kernel=lambda X1, X2: np.exp(
                    -0.5 * scipy.spatial.distance.cdist(
                        X1, X2, 'sqeuclidean'
                    )
                ),
                noise=0.1,
                acquisition='PI'
            )

            bayes_opt1.fit(df_X_train, df_y_train, model=RandomForestClassifier)
            print(bayes_opt1.best_params)

            clf1 = bayes_opt1.get_model()
            clf1.fit(df_X_train, df_y_train)
            print(clf1.score(df_X_test, df_y_test))
            print(matthews_corrcoef(df_y_test, clf1.predict(df_X_test)))
            print((clf1.predict(df_X_test) == df_y_test).value_counts())

            # score

        if 'genetic' in query:
            genetic = Genetic(
                population_size=3,
                offspring_size=3,
                n_generations=3,
                mutation_prob=0.1,
                crossover=lambda x, y: {
                    key: x[key] if np.random.rand() > 0.5 else y[key]
                    for key in x.keys()
                },
                mutation=lambda x: {
                    key: np.clip(
                        int(x[key] + np.random.normal(0, 1)),
                        domain[0][0],
                        domain[0][1]
                    )
                    if key == 'max_depth'
                    else np.clip(
                        int(x[key] + np.random.normal(0, 1)),
                        domain[1][0],
                        domain[1][1]
                    )
                    if key == 'min_samples_split'
                    else np.clip(
                        int(x[key] + np.random.normal(0, 1)),
                        domain[2][0],
                        domain[2][1]
                    )
                    if key == 'min_samples_leaf'
                    else np.clip(
                        int(x[key] + np.random.normal(0, 1)),
                        domain[3][0],
                        domain[3][1]
                    )
                    if key == 'max_features'
                    else np.clip(
                        int(x[key] + np.random.normal(0, 1)),
                        domain[4][0],
                        domain[4][1]
                    )
                    if key == 'max_leaf_nodes'
                    else np.clip(
                        x[key] + np.random.normal(0, 1),
                        domain[5][0],
                        domain[5][1]
                    )
                    if key == 'min_impurity_decrease'
                    else x[key]
                    for key in x.keys()
                },
                random_state=42
            )
            genetic.fit(df_X_train, df_y_train, model=RandomForestClassifier)
            print(genetic.best_params)
            clf2 = genetic.get_model()
            clf2.fit(df_X_train, df_y_train)
            print(clf2.score(df_X_test, df_y_test))
            print(matthews_corrcoef(df_y_test, clf2.predict(df_X_test)))
            print((clf2.predict(df_X_test) == df_y_test).value_counts())

            # score
    