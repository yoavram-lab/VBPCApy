// errpca_pt.cpp (updated pybind11 extension to preserve CSR structure and avoid dense AS)
//
// Semantics (match MATLAB MEX):
// - Compute Err = X - A*S ONLY on the sparsity pattern of X
// - Preserve the exact CSR structure: same indptr/indices length and ordering
// - Store residuals for every stored entry (including exact zeros)
// - Complexity: O(nnz(X) * ncomp), not O(n_rows*n_cols*ncomp)
//
// Python signature:
//   errpca_pt(X_data, X_indices, X_indptr, A, S, numCPU=1) -> dict
// Returns dict containing CSR arrays for Err with SAME indices/indptr as X.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include <Eigen/Dense>
#include <algorithm>
#include <stdexcept>
#include <thread>
#include <vector>

namespace py = pybind11;

static py::dict errpca_pt(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &X_data,
    const py::array_t<int,    py::array::c_style | py::array::forcecast> &X_indices,
    const py::array_t<int,    py::array::c_style | py::array::forcecast> &X_indptr,
    const Eigen::MatrixXd &A,   // (n_rows, ncomp)
    const Eigen::MatrixXd &S,   // (ncomp, n_cols)
    int numCPU
) {
    const auto data_buf   = X_data.request();
    const auto idx_buf    = X_indices.request();
    const auto indptr_buf = X_indptr.request();

    if (indptr_buf.ndim != 1 || indptr_buf.size < 1) {
        throw std::runtime_error("X_indptr must be 1-D with length >= 1.");
    }
    if (idx_buf.ndim != 1 || data_buf.ndim != 1) {
        throw std::runtime_error("X_data and X_indices must be 1-D arrays.");
    }
    if (idx_buf.size != data_buf.size) {
        throw std::runtime_error("X_indices and X_data must have the same length.");
    }

    const int n_rows = static_cast<int>(indptr_buf.size) - 1;
    if (n_rows < 0) {
        throw std::runtime_error("Invalid CSR: n_rows < 0.");
    }
    const int ncomp = static_cast<int>(A.cols());
    const int n_cols = static_cast<int>(S.cols());

    if (A.rows() != n_rows) {
        throw std::runtime_error("A.rows() must equal number of CSR rows.");
    }
    if (S.rows() != ncomp) {
        throw std::runtime_error("S.rows() must equal A.cols().");
    }
    if (n_cols <= 0) {
        throw std::runtime_error("S must have at least one column.");
    }

    auto *x_data   = static_cast<const double *>(data_buf.ptr);
    auto *x_idx    = static_cast<const int *>(idx_buf.ptr);
    auto *x_indptr = static_cast<const int *>(indptr_buf.ptr);

    const int nnz = static_cast<int>(data_buf.size);

    // Validate CSR indptr monotonicity + bounds.
    if (x_indptr[0] != 0 || x_indptr[n_rows] != nnz) {
        throw std::runtime_error("Invalid CSR: indptr[0] must be 0 and indptr[-1] must equal nnz.");
    }
    for (int r = 0; r < n_rows; ++r) {
        if (x_indptr[r] > x_indptr[r + 1]) {
            throw std::runtime_error("Invalid CSR: indptr must be non-decreasing.");
        }
    }

    // Validate indices range
    if (nnz > 0) {
        int max_col = *std::max_element(x_idx, x_idx + nnz);
        int min_col = *std::min_element(x_idx, x_idx + nnz);
        if (min_col < 0 || max_col >= n_cols) {
            throw std::runtime_error("CSR column indices out of range relative to S.cols().");
        }
    }

    // Allocate output data (same length as X_data). Structure is preserved by copying indices/indptr.
    std::vector<double> err_data(static_cast<std::size_t>(nnz));

    // Thread partitioning by rows: each thread writes only within its row slices in err_data.
    int threads = numCPU <= 0 ? 1 : numCPU;
    if (threads > n_rows) threads = n_rows;
    if (threads <= 0) threads = 1;

    auto worker = [&](int row_start, int row_end) {
        for (int r = row_start; r < row_end; ++r) {
            const int start = x_indptr[r];
            const int end   = x_indptr[r + 1];
            for (int p = start; p < end; ++p) {
                const int c = x_idx[p];

                // Compute dot(A[r, :], S[:, c]) without forming A*S
                double res = 0.0;
                for (int k = 0; k < ncomp; ++k) {
                    res += A(r, k) * S(k, c);
                }

                err_data[static_cast<std::size_t>(p)] = x_data[p] - res;
            }
        }
    };

    std::vector<std::thread> pool;
    pool.reserve(static_cast<std::size_t>(threads));

    const int rows_per_thread = n_rows / threads;
    const int remainder = n_rows % threads;

    int cur = 0;
    for (int t = 0; t < threads; ++t) {
        const int extra = (t < remainder) ? 1 : 0;
        const int start = cur;
        const int end   = start + rows_per_thread + extra;
        cur = end;
        pool.emplace_back(worker, start, end);
    }
    for (auto &th : pool) th.join();

    // Build numpy arrays for output; copy indices/indptr verbatim.
    py::array_t<double> out_data(nnz);
    py::array_t<int>    out_indices(nnz);
    py::array_t<int>    out_indptr(n_rows + 1);

    std::memcpy(out_data.mutable_data(), err_data.data(), static_cast<std::size_t>(nnz) * sizeof(double));
    std::memcpy(out_indices.mutable_data(), x_idx, static_cast<std::size_t>(nnz) * sizeof(int));
    std::memcpy(out_indptr.mutable_data(), x_indptr, static_cast<std::size_t>(n_rows + 1) * sizeof(int));

    py::dict result;
    result["data"] = out_data;
    result["indices"] = out_indices;
    result["indptr"] = out_indptr;

    // You can’t recover n_cols from CSR alone; we use S.cols() as authoritative.
    result["shape"] = py::make_tuple(n_rows, n_cols);
    return result;
}

PYBIND11_MODULE(errpca_pt, m) {
    m.doc() = "errpca_pt: CSR-structured sparse reconstruction errors (X - A*S) "
              "preserving the sparsity pattern of X (MATLAB-compatible).";

    m.def(
        "errpca_pt",
        &errpca_pt,
        py::arg("X_data"),
        py::arg("X_indices"),
        py::arg("X_indptr"),
        py::arg("A"),
        py::arg("S"),
        py::arg("numCPU") = 1,
        "Compute sparse matrix of reconstruction errors (X - A*S) on the sparsity "
        "pattern of X, preserving CSR structure."
    );
}
