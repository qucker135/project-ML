import itertools
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.model_selection import cross_val_score
from format_helpers import ndarray_to_params_dict

N_ESTIMATORS = 10

# search space
# max_depth = (9, 11, 1)
# min_samples_split = (2, 21, 2)
# min_samples_leaf = (1, 6, 1)
# max_features = (2, 21, 3)
# max_leaf_nodes = (500, 1001, 50)
# min_impurity_decrease = (0.0, 0.001, 21)
# domain = [
#     (3, 5, 1),
#     (2, 11, 2),
#     (1, 6, 1),
#     (2, 21, 3),
#     (5, 32, 1),
#     (0.0, 0.001, 21)
# ]
domain = [
    (9, 11, 1),
    (2, 11, 2),
    (1, 6, 1),
    (2, 21, 3),
    (500, 1001, 50),
    (0.0, 0.001, 21)
]


def generate_random_params_from_domain(domain) -> np.ndarray:
    return np.array(
        [
            np.random.randint(low, high)
            if type(low) == int
            else np.random.randint(0, step) / (step - 1) * (high - low) + low
            if type(low) == float
            else ValueError(f"Unsupported type {type(low)}")
            for (low, high, step) in domain
        ],
        dtype=np.float64,
    )


def iterate_through_whole_domain(domain) -> np.ndarray:
    domain_iters = [
        range(low, high, step)
        if type(low) == int
        else map(lambda x: low + (high - low) * x / (step - 1), range(step))
        if type(low) == float
        else ValueError(f"Unsupported type {type(low)}")
        for (low, high, step) in domain
    ]
    return itertools.product(*domain_iters)


def target_function(
        params_vector: np.ndarray,
        X,
        y,
        model=RandomForestClassifier
        ) -> float:
    params_dict = ndarray_to_params_dict(params_vector)
    params_dict["random_state"] = 42
    params_dict["n_estimators"] = N_ESTIMATORS
    print(f"{params_dict=}")
    clf = model(**params_dict)
    results = cross_val_score(
        clf,
        X,
        y,
        cv=5,
        scoring=make_scorer(matthews_corrcoef)
    )
    print(f"{results=}")
    return results.mean()
