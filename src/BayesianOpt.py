from sklearn.ensemble import RandomForestClassifier
from GPRegressor import GPRegressor
import numpy as np
import scipy
from format_helpers import ndarray_to_params_dict
from settings import N_ESTIMATORS, domain
from settings import generate_random_params_from_domain
from settings import iterate_through_whole_domain, target_function


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
