// errpca_pt.cpp
//
// Compute a sparse matrix of reconstruction errors for probabilistic PCA
// using the sparse structure of X and optional multithreading.
//
// Python signature (via pybind11):
//   errpca_pt(X_data, X_indices, X_indptr, A, S, numCPU=1) -> dict
//
// Inputs (CSR of X and dense factors A, S):
//   - X_data   : 1D array of nonzero values of X (double)
//   - X_indices: 1D array of column indices for each nonzero (int32)
//   - X_indptr : 1D array of row pointer offsets (int32), length = n_rows + 1
//   - A        : Eigen::MatrixXd of shape (n_rows, n_components)
//   - S        : Eigen::MatrixXd of shape (n_components, n_cols)
//   - numCPU   : requested number of threads (>= 1)
//
// Output (Python dict of CSR of Err = X - A*S, with sparsity pattern of X):
//   {
//       "data":   1D numpy array (double),
//       "indices":1D numpy array (int32),
//       "indptr": 1D numpy array (int32),
//       "shape":  (n_rows, n_cols)
//   }
//
// Build (example):
//   python3 setup.py build_ext --inplace

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include <Eigen/Sparse>
#include <Eigen/Dense>

#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace py = pybind11;

// -----------------------------------------------------------------------------
// Type aliases
// -----------------------------------------------------------------------------

using SpMat   = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using Triplet = Eigen::Triplet<double>;

// -----------------------------------------------------------------------------
// CSR -> Eigen SparseMatrix conversion
// -----------------------------------------------------------------------------

static SpMat csr_to_eigen(
    int n_rows,
    int n_cols,
    const std::vector<int> &indptr,
    const std::vector<int> &indices,
    const std::vector<double> &data
) {
    std::vector<Triplet> triplet_list;
    triplet_list.reserve(data.size());

    for (int row = 0; row < n_rows; ++row) {
        for (int idx = indptr[row]; idx < indptr[row + 1]; ++idx) {
            const int    col = indices[idx];
            const double val = data[idx];
            triplet_list.emplace_back(row, col, val);
        }
    }

    SpMat mat(n_rows, n_cols);
    mat.setFromTriplets(triplet_list.begin(), triplet_list.end());
    return mat;
}

// -----------------------------------------------------------------------------
// Eigen SparseMatrix -> CSR conversion
// -----------------------------------------------------------------------------

static void eigen_to_csr(
    const SpMat &mat,
    std::vector<double> &data,
    std::vector<int> &indices,
    std::vector<int> &indptr
) {
    const int n_rows = static_cast<int>(mat.rows());

    data.reserve(mat.nonZeros());
    indices.reserve(mat.nonZeros());
    indptr.assign(n_rows + 1, 0);

    for (int outer = 0; outer < mat.outerSize(); ++outer) {
        for (SpMat::InnerIterator it(mat, outer); it; ++it) {
            data.push_back(it.value());
            indices.push_back(it.col());
            indptr[it.row() + 1] += 1;
        }
    }

    // Convert counts -> cumulative offsets.
    for (int i = 0; i < n_rows; ++i) {
        indptr[i + 1] += indptr[i];
    }
}

// -----------------------------------------------------------------------------
// Core computation: Err = X - A*S, sparsity pattern of X
// -----------------------------------------------------------------------------

static py::dict errpca_pt(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &X_data,
    const py::array_t<int,    py::array::c_style | py::array::forcecast> &X_indices,
    const py::array_t<int,    py::array::c_style | py::array::forcecast> &X_indptr,
    const Eigen::MatrixXd &A,
    const Eigen::MatrixXd &S,
    int numCPU
) {
    // -------------------------------------------------------------------------
    // Extract CSR components from Python arrays
    // -------------------------------------------------------------------------
    const auto X_data_buf   = X_data.request();
    const auto X_indices_buf = X_indices.request();
    const auto X_indptr_buf  = X_indptr.request();

    if (X_indptr_buf.size < 1) {
        throw std::runtime_error("X_indptr must have length >= 1.");
    }

    std::vector<double> X_data_vec(
        static_cast<const double *>(X_data_buf.ptr),
        static_cast<const double *>(X_data_buf.ptr) + X_data_buf.size
    );

    std::vector<int> X_indices_vec(
        static_cast<const int *>(X_indices_buf.ptr),
        static_cast<const int *>(X_indices_buf.ptr) + X_indices_buf.size
    );

    std::vector<int> X_indptr_vec(
        static_cast<const int *>(X_indptr_buf.ptr),
        static_cast<const int *>(X_indptr_buf.ptr) + X_indptr_buf.size
    );

    const int n_rows = static_cast<int>(X_indptr_vec.size()) - 1;
    const int n_cols = static_cast<int>(S.cols());

    if (n_rows < 0) {
        throw std::runtime_error("Invalid CSR structure: n_rows < 0.");
    }

    // -------------------------------------------------------------------------
    // Dimension checks
    // -------------------------------------------------------------------------
    if (A.rows() != n_rows) {
        throw std::runtime_error("Matrix A must have the same number of rows as X.");
    }
    if (A.cols() != S.rows()) {
        throw std::runtime_error("Number of columns in A must match number of rows in S.");
    }
    if (S.cols() <= 0) {
        throw std::runtime_error("Matrix S must have at least one column.");
    }

    // Ensure CSR column indices are within [0, n_cols).
    if (!X_indices_vec.empty()) {
        const auto max_it = std::max_element(X_indices_vec.begin(), X_indices_vec.end());
        if (*max_it >= n_cols || *max_it < 0) {
            throw std::runtime_error(
                "CSR column indices for X are out of range relative to S.cols()."
            );
        }
    }

    // -------------------------------------------------------------------------
    // Edge case: no rows
    // -------------------------------------------------------------------------
    if (n_rows == 0) {
        py::dict result;
        result["data"]   = py::array_t<double>(0);
        result["indices"] = py::array_t<int>(0);
        result["indptr"]  = py::array_t<int>(1);  // single 0
        {
            auto indptr = result["indptr"].cast<py::array_t<int>>();
            auto buf = indptr.request();
            static_cast<int *>(buf.ptr)[0] = 0;
        }
        result["shape"] = py::make_tuple(0, n_cols);
        return result;
    }

    // -------------------------------------------------------------------------
    // Convert CSR X -> Eigen sparse, allocate Err
    // -------------------------------------------------------------------------
    SpMat X = csr_to_eigen(n_rows, n_cols, X_indptr_vec, X_indices_vec, X_data_vec);

    SpMat Err(n_rows, n_cols);
    Err.reserve(X.nonZeros());

    // Dense product A*S (n_rows x n_cols)
    const Eigen::MatrixXd AS = A * S;

    // -------------------------------------------------------------------------
    // Multithreading setup
    // -------------------------------------------------------------------------
    int threads_requested = numCPU;
    if (threads_requested <= 0) {
        threads_requested = 1;
    }
    if (threads_requested > n_rows) {
        threads_requested = n_rows;
    }
    if (threads_requested <= 0) {
        threads_requested = 1;
    }

    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(threads_requested));

    std::mutex triplet_mutex;
    std::vector<Triplet> triplet_list;
    triplet_list.reserve(X.nonZeros());

    auto worker = [&](int start_row, int end_row) {
        std::vector<Triplet> local_triplets;
        for (int row = start_row; row < end_row; ++row) {
            for (SpMat::InnerIterator it(X, row); it; ++it) {
                const int    col   = it.col();
                const double x_val = it.value();
                const double as_val = AS(row, col);
                const double err   = x_val - as_val;
                if (err != 0.0) {  // store only non-zero errors
                    local_triplets.emplace_back(row, col, err);
                }
            }
        }
        if (!local_triplets.empty()) {
            std::lock_guard<std::mutex> lock(triplet_mutex);
            triplet_list.insert(
                triplet_list.end(),
                local_triplets.begin(),
                local_triplets.end()
            );
        }
    };

    const int rows_per_thread = n_rows / threads_requested;
    const int remainder       = n_rows % threads_requested;

    int current_row = 0;
    for (int t = 0; t < threads_requested; ++t) {
        const int start_row = current_row;
        const int extra     = (t < remainder) ? 1 : 0;
        const int end_row   = start_row + rows_per_thread + extra;
        current_row         = end_row;
        threads.emplace_back(worker, start_row, end_row);
    }

    for (auto &th : threads) {
        th.join();
    }

    // -------------------------------------------------------------------------
    // Build Err sparse matrix from triplets and convert back to CSR
    // -------------------------------------------------------------------------
    Err.setFromTriplets(triplet_list.begin(), triplet_list.end());

    std::vector<double> Err_data;
    std::vector<int>    Err_indices;
    std::vector<int>    Err_indptr;
    eigen_to_csr(Err, Err_data, Err_indices, Err_indptr);

    // Wrap CSR components into numpy arrays (zero-copy on data)
    py::array_t<double> Err_data_py(
        Err_data.size(),
        Err_data.empty() ? nullptr : Err_data.data()
    );
    py::array_t<int> Err_indices_py(
        Err_indices.size(),
        Err_indices.empty() ? nullptr : Err_indices.data()
    );
    py::array_t<int> Err_indptr_py(
        Err_indptr.size(),
        Err_indptr.empty() ? nullptr : Err_indptr.data()
    );

    py::dict result;
    result["data"]    = Err_data_py;
    result["indices"] = Err_indices_py;
    result["indptr"]  = Err_indptr_py;
    result["shape"]   = py::make_tuple(n_rows, n_cols);

    return result;
}

// -----------------------------------------------------------------------------
// pybind11 module definition
// -----------------------------------------------------------------------------

PYBIND11_MODULE(errpca_pt, m) {
    m.doc() = "errpca_pt: compute sparse reconstruction errors (X - A*S) "
              "with sparsity pattern of X.";

    m.def(
        "errpca_pt",
        &errpca_pt,
        py::arg("X_data"),
        py::arg("X_indices"),
        py::arg("X_indptr"),
        py::arg("A"),
        py::arg("S"),
        py::arg("numCPU") = 1,
        "Compute sparse matrix of reconstruction errors (X - A*S) using the "
        "sparsity pattern of X."
    );
}
