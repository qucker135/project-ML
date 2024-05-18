from pathlib import Path
import pandas as pd
import numpy as np
import scipy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from BayesianOpt import BayesOpt

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

    clf = bayes_opt1.get_model()
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    print(matthews_corrcoef(y_test, clf.predict(X_test)))
    print((clf.predict(X_test) == y_test).value_counts())
