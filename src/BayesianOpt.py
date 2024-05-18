from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.model_selection import cross_val_score
from GPRegressor import GPRegressor
import numpy as np
import scipy
import itertools
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


class BayesOpt:
    def __init__(
            self,
            init_guesses: int,
            n_iter: int,
            kernel,
            noise: float,
            acquisition: str = "UCB"
            ):
        self.init_guesses = init_guesses
        self.n_iter = n_iter
        self.gp = GPRegressor(kernel, noise)
        self.best_params = None
        self.acquisition = acquisition

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            model=RandomForestClassifier
            ) -> None:
        params_vectors = []
        params_scores = []

        best_params = None
        best_score = -np.inf

        for _ in range(self.init_guesses):
            params_vector = generate_random_params_from_domain(domain)
            score = target_function(params_vector, X, y, model)
            params_vectors.append(params_vector)
            params_scores.append(score)
            if score > best_score:
                best_score = score
                best_params = params_vector

        print("Initial guesses:")
        for i, j in zip(params_vectors, params_scores):
            print(f"{i}:, {j}")

        for dbg_i in range(self.n_iter):
            print(f"iteration {dbg_i}")
            self.gp.fit(params_vectors, params_scores)
            # optimize acquisition function
            tmp_domain_vector = None
            tmp_best_score = -np.inf
            for domain_param_vector in iterate_through_whole_domain(domain):
                if any(
                        np.array_equal(domain_param_vector, x)
                        for x in params_vectors
                        ):
                    continue
                mu, cov = self.gp.predict([domain_param_vector])
                if self.acquisition == "UCB":
                    tmp_score = mu + 0.5 * np.sqrt(np.diag(cov))
                elif self.acquisition == "PI":
                    tmp_score = scipy.stats.norm.cdf(
                        (mu - np.max(
                            params_scores
                        ) - 1E-3) / (np.sqrt(np.diag(cov)) + 1E-9)
                    )
                else:
                    raise ValueError(
                        f"Unsupported acquisition function {self.acquisition}"
                    )
                if tmp_score > tmp_best_score:
                    tmp_best_score = tmp_score
                    tmp_domain_vector = domain_param_vector

            print(f"Best domain vector: {tmp_domain_vector}")

            params_vector = tmp_domain_vector
            score = target_function(params_vector, X, y, model)

            print(f"{params_vector=}")
            print(f"{score=}")
            params_vectors.append(params_vector)
            params_scores.append(score)
            print(f"{params_vectors=}")
            print(f"{params_scores=}")
            if score > best_score:
                print(f"New best score: {score}")
                best_score = score
                best_params = params_vector

            print(f"Best params: {best_params}")
            print(f"Best score: {best_score}")
            print(f"self best params: {self.best_params=}")

        self.best_params = ndarray_to_params_dict(best_params)
        self.best_params["n_estimators"] = N_ESTIMATORS

    def get_model(self) -> RandomForestClassifier:
        if self.best_params is None:
            raise ValueError("Model has not been trained yet")
        return RandomForestClassifier(**self.best_params)
