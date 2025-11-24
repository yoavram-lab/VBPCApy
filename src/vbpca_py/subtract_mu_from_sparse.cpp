// subtract_mu_from_sparse.cpp
//
// Subtract a row-wise mean vector (Mu) from a sparse matrix X in CSR format,
// preserving the sparsity structure and replacing exact zeros with a small
// constant EPS for numerical stability.
//
// Python signature (via pybind11):
//   subtract_mu_from_sparse(data, indices, indptr, shape, Mu) -> np.ndarray
//
// Inputs:
//   - data    : 1D array of nonzero values of X (double), length = nnz
//   - indices : 1D array of column indices for each nonzero (int), length = nnz
//   - indptr  : 1D array of row pointer offsets (int), length = n_rows + 1
//   - shape   : Python tuple (n_rows, n_cols)
//   - Mu      : 1D array of length >= n_rows (double)
//
// Output:
//   - 1D numpy array of length nnz (double), corresponding to updated data
//     values after subtracting Mu[row] from each stored entry in that row.
//     Exact zeros are replaced by EPS.
//
// Build is configured in setup.py as extension "vbpca_py.subtract_mu_from_sparse".

#include <cmath>
#include <stdexcept>
#include <thread>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

constexpr double EPS = 1e-15;

// -----------------------------------------------------------------------------
// Core implementation
// -----------------------------------------------------------------------------

py::array_t<double> subtract_mu_from_sparse(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &mxX_data_array,
    const py::array_t<int,    py::array::c_style | py::array::forcecast> &mxX_indices_array,
    const py::array_t<int,    py::array::c_style | py::array::forcecast> &mxX_indptr_array,
    const py::tuple &mxX_shape,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &mxMu_array
) {
    // Extract shape
    if (mxX_shape.size() != 2) {
        throw std::invalid_argument("SUBTRACT_MU: shape must be a 2-tuple (n_rows, n_cols).");
    }
    const int n_rows = mxX_shape[0].cast<int>();
    const int n_cols = mxX_shape[1].cast<int>();
    if (n_rows < 0 || n_cols < 0) {
        throw std::invalid_argument("SUBTRACT_MU: n_rows and n_cols must be non-negative.");
    }

    // Buffer views
    const auto data_buf    = mxX_data_array.request();
    const auto indices_buf = mxX_indices_array.request();
    const auto indptr_buf  = mxX_indptr_array.request();
    const auto mu_buf      = mxMu_array.request();

    const std::size_t nnz = static_cast<std::size_t>(data_buf.size);

    // Basic consistency checks
    if (indices_buf.size != static_cast<ssize_t>(nnz)) {
        throw std::invalid_argument(
            "SUBTRACT_MU: X_data and X_indices must have the same length (nnz)."
        );
    }

    if (n_rows == 0) {
        // For an empty matrix, indptr should be length 1 and nnz must be 0.
        if (indptr_buf.size != 1) {
            throw std::invalid_argument(
                "SUBTRACT_MU: indptr size must be 1 when n_rows == 0."
            );
        }
        if (nnz != 0) {
            throw std::invalid_argument(
                "SUBTRACT_MU: nnz must be 0 when n_rows == 0."
            );
        }
        // Just return an empty array.
        return py::array_t<double>(0);
    }

    if (indptr_buf.size != static_cast<ssize_t>(n_rows + 1)) {
        throw std::invalid_argument(
            "SUBTRACT_MU: indptr size must be n_rows + 1."
        );
    }

    // Number of nonzeros from CSR view
    const auto *indptr_ptr = static_cast<const int *>(indptr_buf.ptr);
    const int nnz_from_csr = indptr_ptr[n_rows];
    if (nnz_from_csr < 0) {
        throw std::invalid_argument(
            "SUBTRACT_MU: last entry of indptr (nnz) must be non-negative."
        );
    }
    if (static_cast<std::size_t>(nnz_from_csr) != nnz) {
        throw std::invalid_argument(
            "SUBTRACT_MU: nnz inferred from indptr does not match data length."
        );
    }

    // Mu length check
    if (mu_buf.size < n_rows) {
        throw std::invalid_argument(
            "SUBTRACT_MU: Mu vector length does not match the number of rows in X."
        );
    }

    // Column indices bounds check
    const auto *indices_ptr = static_cast<const int *>(indices_buf.ptr);
    for (std::size_t k = 0; k < nnz; ++k) {
        const int col = indices_ptr[k];
        if (col < 0 || col >= n_cols) {
            throw std::invalid_argument(
                "SUBTRACT_MU: CSR column indices are out of range for the given shape."
            );
        }
    }

    // Accessors
    auto X_data = mxX_data_array.unchecked<1>();
    auto X_indptr = mxX_indptr_array.unchecked<1>();
    auto Mu = mxMu_array.unchecked<1>();

    // Prepare output array and mutable view
    py::array_t<double> Xout_array(nnz);
    auto Xout = Xout_array.mutable_unchecked<1>();

    if (nnz == 0) {
        // Nothing to do; just return zero-length array
        return Xout_array;
    }

    // -------------------------------------------------------------------------
    // Multithreading over rows
    // -------------------------------------------------------------------------
    // Choose number of threads based on hardware and number of rows.
    unsigned int hw_threads = std::thread::hardware_concurrency();
    int num_threads = hw_threads > 0 ? static_cast<int>(hw_threads) : 1;
    if (num_threads > n_rows) {
        num_threads = n_rows;
    }
    if (num_threads < 1) {
        num_threads = 1;
    }

    auto worker = [&](int row_start, int row_end) {
        for (int row = row_start; row < row_end; ++row) {
            const double mu_row = Mu(row);
            const int start_idx = X_indptr(row);
            const int end_idx   = X_indptr(row + 1);
            for (int idx = start_idx; idx < end_idx; ++idx) {
                const std::size_t p = static_cast<std::size_t>(idx);
                double val = X_data(p) - mu_row;
                if (val == 0.0) {
                    val = EPS;
                }
                Xout(p) = val;
            }
        }
    };

    if (num_threads == 1) {
        worker(0, n_rows);
    } else {
        std::vector<std::thread> threads;
        threads.reserve(static_cast<std::size_t>(num_threads));

        const int rows_per_thread = n_rows / num_threads;
        const int remainder       = n_rows % num_threads;

        int current_row = 0;
        for (int t = 0; t < num_threads; ++t) {
            const int start_row = current_row;
            const int extra     = (t < remainder) ? 1 : 0;
            const int end_row   = start_row + rows_per_thread + extra;
            current_row         = end_row;
            threads.emplace_back(worker, start_row, end_row);
        }

        for (auto &th : threads) {
            th.join();
        }
    }

    return Xout_array;
}

// -----------------------------------------------------------------------------
// pybind11 module definition
// -----------------------------------------------------------------------------

PYBIND11_MODULE(subtract_mu_from_sparse, m) {
    m.doc() = "Subtract Mu from sparse matrix X using C++ and pybind11.";

    m.def(
        "subtract_mu_from_sparse",
        &subtract_mu_from_sparse,
        py::arg("data"),
        py::arg("indices"),
        py::arg("indptr"),
        py::arg("shape"),
        py::arg("Mu"),
        "Subtract a row-wise mean vector Mu from a CSR matrix X.\n\n"
        "Parameters\n"
        "----------\n"
        "data : array_like of float64\n"
        "    Nonzero values of X (length nnz).\n"
        "indices : array_like of int32\n"
        "    Column indices for each nonzero (length nnz).\n"
        "indptr : array_like of int32\n"
        "    Row pointer array of length n_rows + 1.\n"
        "shape : tuple\n"
        "    (n_rows, n_cols) shape of X.\n"
        "Mu : array_like of float64\n"
        "    Mean vector of length >= n_rows.\n\n"
        "Returns\n"
        "-------\n"
        "out_data : np.ndarray of float64\n"
        "    Updated nonzero values after subtracting Mu[row] from each\n"
        "    nonzero in that row; exact zeros are replaced by EPS."
    );
}
