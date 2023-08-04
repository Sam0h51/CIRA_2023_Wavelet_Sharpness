import numpy as np
from skimage import filters


def mse(X, T):
    """Mean Squared Error"""
    return np.mean((X - T) ** 2)


def mae(X, T):
    """Mean Absolute Error"""
    return np.mean(np.abs((X - T)))


def rmse(X, T):
    """Root Mean Squared Error"""
    return np.sqrt(mse(X, T))


def grad(X, T):
    """Average Magnitude of the Gradient

    Edge magnitude is computed as:
        sqrt(Gx^2 + Gy^2)
    """
    def _f(x): return np.mean(filters.sobel(x))
    return (_f(X), _f(T))


######### Default Metrics #########
metric_f = {
    'mse': mse,
    'mae': mae,
    'rmse': rmse,
    'grad': grad,
}


def compute_all_metrics(X, T) -> dict:
    """Compute all evaluation metrics."""
    results = dict()
    for metric, f in metric_f.items():
        try:
            results[metric] = f(X, T)
        except Exception as e:
            print(f'Failed to compute {metric}: {e}')
    return results


def compute_metric(X, T, metric: str):
    """Compute specified evaluation metric"""
    f = metric_f.get(metric)
    if f is None:
        raise ValueError(f'Unknown metric name: {metric}')
    return f(X, T)


if __name__ == '__main__':
    from skimage.data import camera
    X = camera()
    T = np.fliplr(X)

    results = compute_all_metrics(X, T)
    for metric, result in results.items():
        print(f'{metric}: {result}')
