import numpy as np
import scipy
import typing


class GPRegressor:
    def __init__(self, kernel, noise=0.1):
        self.kernel = kernel
        self.noise = noise

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        self.K = self.kernel(X, X) + np.eye(len(X)) * (self.noise + 1E-2) ** 2

    def predict(self, X: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        K_s = self.kernel(self.X, X)
        K_ss = self.kernel(X, X)
        K_inv = np.linalg.inv(self.K)

        mu = K_s.T @ K_inv @ self.y
        cov = K_ss - K_s.T @ K_inv @ K_s

        return mu, cov


if __name__ == "__main__":
    def kernel(X1, X2):
        return np.exp(
            -0.5 * scipy.spatial.distance.cdist(X1, X2, 'sqeuclidean')
        )

    gp = GPRegressor(kernel)

    X = np.random.rand(10, 1)
    y = np.sin(X).ravel()

    gp.fit(X, y)

    X_test = np.linspace(0, 1, 100).reshape(-1, 1)
    y_pred, cov = gp.predict(X_test)

    import matplotlib.pyplot as plt
    plt.plot(X_test, y_pred)
    plt.fill_between(
        X_test.ravel(),
        y_pred - np.sqrt(np.diag(cov)),
        y_pred + np.sqrt(np.diag(cov)),
        alpha=0.5
    )
    plt.scatter(X, y)
    plt.show()
