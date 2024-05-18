import numpy as np
import typing


def params_dict_to_ndarray(params: typing.Dict[str, typing.Any]) -> np.ndarray:
    return np.array([
        params["max_depth"],
        params["min_samples_split"],
        params["min_samples_leaf"],
        params["max_features"],
        params["max_leaf_nodes"],
        params["min_impurity_decrease"]
    ], dtype=np.float64)


def ndarray_to_params_dict(arr: np.ndarray) -> typing.Dict[str, typing.Any]:
    return {
        "max_depth": int(arr[0]),
        "min_samples_split": int(arr[1]),
        "min_samples_leaf": int(arr[2]),
        "max_features": int(arr[3]),
        "max_leaf_nodes": int(arr[4]),
        "min_impurity_decrease": arr[5]
    }


if __name__ == "__main__":
    params = {
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": 4,
        "max_leaf_nodes": 10,
        "min_impurity_decrease": 0.1
    }
    arr = params_dict_to_ndarray(params)
    print(arr)
    print(ndarray_to_params_dict(arr))
