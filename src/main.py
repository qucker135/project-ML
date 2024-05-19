from pathlib import Path
import pandas as pd
import numpy as np
import scipy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from BayesianOpt import BayesOpt
from Genetic import Genetic
from settings import domain

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / 'datasets' / 'nasa_cleaned.csv'

if __name__ == '__main__':
    df = pd.read_csv(DATASET_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop('Hazardous', axis=1),
        df['Hazardous'],
        test_size=0.2,
        random_state=42
    )

    query = input("What do you want to do? (bayes/genetic):")

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

        bayes_opt1.fit(X_train, y_train, model=RandomForestClassifier)
        print(bayes_opt1.best_params)

        clf1 = bayes_opt1.get_model()
        clf1.fit(X_train, y_train)
        print(clf1.score(X_test, y_test))
        print(matthews_corrcoef(y_test, clf1.predict(X_test)))
        print((clf1.predict(X_test) == y_test).value_counts())

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
        genetic.fit(X_train, y_train, model=RandomForestClassifier)
        print(genetic.best_params)
        clf2 = genetic.get_model()
        clf2.fit(X_train, y_train)
        print(clf2.score(X_test, y_test))
        print(matthews_corrcoef(y_test, clf2.predict(X_test)))
        print((clf2.predict(X_test) == y_test).value_counts())
