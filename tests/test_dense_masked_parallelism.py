import numpy as np

from vbpca_py import dense_update_kernels as duk


def _fixture():
    rng = np.random.default_rng(42)
    n_features = 5
    n_samples = 4
    n_components = 3

    x_data = rng.normal(size=(n_features, n_samples))
    mask = (rng.random(size=(n_features, n_samples)) > 0.25).astype(float)
    loadings = rng.normal(scale=0.5, size=(n_features, n_components))
    scores = rng.normal(scale=0.5, size=(n_components, n_samples))

    av = np.stack([np.eye(n_components) * 0.05 for _ in range(n_features)])
    sv = np.stack([np.eye(n_components) * 0.02 for _ in range(n_samples)])
    prior_prec = np.eye(n_components) * 0.1
    return x_data, mask, loadings, scores, av, sv, prior_prec


def test_score_update_dense_masked_parallel_parity() -> None:
    x_data, mask, loadings, _scores, av, _sv, _prior_prec = _fixture()

    result = duk.score_update_dense_masked_nopattern(
        x_data=x_data,
        mask=mask,
        loadings=loadings,
        loading_covariances=av,
        noise_var=0.5,
        return_covariances=True,
    )

    assert result["scores"].shape == (loadings.shape[1], x_data.shape[1])
    assert result["score_covariances"].shape == (
        x_data.shape[1],
        loadings.shape[1],
        loadings.shape[1],
    )
    assert np.isfinite(result["scores"]).all()
    assert np.isfinite(result["score_covariances"]).all()


def test_loadings_update_dense_masked_parallel_parity() -> None:
    x_data, mask, _loadings, scores, _av, sv, prior_prec = _fixture()

    result = duk.loadings_update_dense_masked_nopattern(
        x_data=x_data,
        mask=mask,
        scores=scores,
        score_covariances=sv,
        prior_prec=prior_prec,
        noise_var=0.5,
        return_covariances=True,
    )

    assert result["loadings"].shape == (x_data.shape[0], scores.shape[0])
    assert result["loading_covariances"].shape == (
        x_data.shape[0],
        scores.shape[0],
        scores.shape[0],
    )
    assert np.isfinite(result["loadings"]).all()
    assert np.isfinite(result["loading_covariances"]).all()
