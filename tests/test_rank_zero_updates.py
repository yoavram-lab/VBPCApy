"""Empty native updates must be safe with supported older Eigen headers."""

import pytest
from test_rank_zero_rotation import _run_isolated


@pytest.mark.parametrize(
    "name",
    [
        "score_update_dense_no_av",
        "loadings_update_dense_no_sv",
        "score_update_dense_masked_nopattern",
        "loadings_update_dense_masked_nopattern",
        "score_update_sparse_nopattern",
        "loadings_update_sparse_nopattern",
    ],
)
def test_rank_zero_update_shapes_and_validation_in_subprocess(name):
    _run_isolated(
        """
        import sys
        from itertools import product
        import numpy as np
        from scipy import sparse
        from vbpca_py import dense_update_kernels as dense
        from vbpca_py import sparse_update_kernels as sparse_kernels

        name = sys.argv[1]
        is_score = name.startswith('score_')
        is_sparse = 'sparse' in name
        is_masked = 'masked' in name
        function = getattr(sparse_kernels if is_sparse else dense, name)
        features, samples = 3, 5
        x = np.arange(features * samples, dtype=float).reshape(features, samples)
        loadings = np.empty((features, 0))
        scores = np.empty((0, samples))
        prior = np.empty((0, 0))
        observed = np.ones_like(x, dtype=bool)
        observed[0, 2] = False
        table = sparse.coo_matrix(
            (np.arange(5, dtype=float), ([0, 1, 2, 0, 2], [0, 1, 2, 3, 4])),
            shape=x.shape,
        )
        table = table.tocsc() if is_score else table.tocsr()
        covariance_name = 'loading_covariances' if is_score else 'score_covariances'
        bank_count = features if is_score else samples
        output_name = 'scores' if is_score else 'loadings'
        output_shape = (0, samples) if is_score else (features, 0)
        covariance_key = ('score_' if is_score else 'loading_') + (
            'covariances' if is_sparse or is_masked else 'covariance'
        )
        covariance_shape = (
            (samples if is_score else features, 0, 0)
            if is_sparse or is_masked else (0, 0)
        )
        varied = is_sparse or is_masked
        for return_cov, bank, threads, complement, numeric_mask in product(
            (False, True), (False, True) if varied else (False,),
            (1, 3) if varied else (1,),
            (False, True) if is_masked else (False,),
            (False, True) if is_masked else (False,),
        ):
            kwargs = {'noise_var': 2.0}
            if is_sparse:
                kwargs.update(x_data=table.data, x_indices=table.indices,
                              x_indptr=table.indptr)
            else:
                kwargs['x_data'] = x
            kwargs['loadings' if is_score else 'scores'] = (
                loadings if is_score else scores
            )
            if not is_score:
                kwargs['prior_prec'] = prior
            if varied:
                kwargs[covariance_name] = np.empty((bank_count, 0, 0)) if bank else None
                kwargs['return_covariances'] = return_cov
                kwargs['num_cpu'] = threads
            else:
                kwargs['return_covariance'] = return_cov
            if is_masked:
                kwargs['mask'] = observed.astype(float) if numeric_mask else observed
                kwargs['use_complement'] = complement
            before = {}
            for key, value in kwargs.items():
                if isinstance(value, np.ndarray):
                    value.setflags(write=False)
                    before[key] = value.copy()
            print(name, return_cov, bank, threads, complement, numeric_mask, flush=True)
            result = function(**kwargs)
            expected_keys = (
                {output_name, covariance_key} if return_cov else {output_name}
            )
            assert set(result) == expected_keys
            assert result[output_name].shape == output_shape
            for value in result.values():
                assert value.dtype == np.float64
                assert value.size == 0
            if return_cov:
                assert result[covariance_key].shape == covariance_shape
            for key, value in before.items():
                np.testing.assert_array_equal(kwargs[key], value)

        def refuse(changes):
            try:
                function(**{**kwargs, **changes})
            except ValueError:
                return
            raise AssertionError(f'rank zero bypassed validation: {changes.keys()}')

        for invalid in (0.0, -1.0, float('nan'), float('inf')):
            refuse({'noise_var': invalid})
        if not is_score:
            refuse({'prior_prec': np.eye(1)})
        if varied:
            refuse({covariance_name: np.empty((bank_count, 0, 1))})
        if is_sparse:
            bad_indices = table.indices.copy()
            bad_indices[0] = features if is_score else samples
            refuse({'x_indices': bad_indices})
            bad_pointers = table.indptr.copy()
            bad_pointers[-1] += 1
            refuse({'x_indptr': bad_pointers})
        else:
            refuse({'loadings': np.empty((features + 1, 0))} if is_score else
                   {'scores': np.empty((0, samples + 1))})
        if is_masked:
            refuse({'mask': np.ones((features, samples + 1), dtype=bool)})
        """,
        name,
    )
