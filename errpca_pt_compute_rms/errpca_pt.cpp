// errpca_pt.cpp

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>
#include <thread>
#include <mutex>

namespace py = pybind11;

// Type definitions for convenience
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpMat;
typedef Eigen::Triplet<double> Triplet;

// Function to convert CSR components to Eigen SparseMatrix
SpMat csr_to_eigen(int n_rows, int n_cols, const std::vector<int>& indptr,
                   const std::vector<int>& indices, const std::vector<double>& data) {
    std::vector<Triplet> tripletList;
    tripletList.reserve(data.size());

    for (int row = 0; row < n_rows; ++row) {
        for (int idx = indptr[row]; idx < indptr[row + 1]; ++idx) {
            int col = indices[idx];
            double val = data[idx];
            tripletList.emplace_back(row, col, val);
        }
    }

    SpMat mat(n_rows, n_cols);
    mat.setFromTriplets(tripletList.begin(), tripletList.end());
    return mat;
}

// Function to convert Eigen SparseMatrix to CSR components
void eigen_to_csr(const SpMat& mat, std::vector<double>& data,
                  std::vector<int>& indices, std::vector<int>& indptr) {
    data.reserve(mat.nonZeros());
    indices.reserve(mat.nonZeros());
    indptr.resize(mat.rows() + 1, 0);

    for (int k = 0; k < mat.outerSize(); ++k) {
        for (SpMat::InnerIterator it(mat, k); it; ++it) {
            data.push_back(it.value());
            indices.push_back(it.col());
            indptr[it.row() + 1]++;
        }
    }

    // Cumulative sum for indptr
    for (int i = 0; i < mat.rows(); ++i) {
        indptr[i + 1] += indptr[i];
    }
}

// Function to compute errpca_pt
py::dict errpca_pt(const py::array_t<double, py::array::c_style | py::array::forcecast>& X_data,
                   const py::array_t<int, py::array::c_style | py::array::forcecast>& X_indices,
                   const py::array_t<int, py::array::c_style | py::array::forcecast>& X_indptr,
                   const Eigen::MatrixXd& A,
                   const Eigen::MatrixXd& S,
                   int numCPU = 1) {
    // Extract CSR components
    std::vector<double> X_data_vec(X_data.size());
    std::copy(X_data.data(), X_data.data() + X_data.size(), X_data_vec.begin());

    std::vector<int> X_indices_vec(X_indices.size());
    std::copy(X_indices.data(), X_indices.data() + X_indices.size(), X_indices_vec.begin());

    std::vector<int> X_indptr_vec(X_indptr.size());
    std::copy(X_indptr.data(), X_indptr.data() + X_indptr.size(), X_indptr_vec.begin());

    // Determine the shape
    int n_rows = X_indptr_vec.size() - 1;
    int n_cols = S.cols();  // Use the number of columns in S

    // Convert CSR to Eigen SparseMatrix
    SpMat X = csr_to_eigen(n_rows, n_cols, X_indptr_vec, X_indices_vec, X_data_vec);

    // Check matrix dimensions
    if (A.rows() != n_rows) {
        throw std::runtime_error("Matrix A must have the same number of rows as X.");
    }
    if (A.cols() != S.rows()) {
        throw std::runtime_error("Number of columns in A must match number of rows in S.");
    }
    if (S.cols() != n_cols) {
        throw std::runtime_error("Number of columns in S must match number of columns in X.");
    }

    // Prepare the output sparse matrix
    SpMat Err(n_rows, n_cols);
    Err.reserve(X.nonZeros());

    // Compute A * S
    Eigen::MatrixXd AS = A * S;  // Shape (n_rows x n_cols)

    // Multi-threading setup
    std::vector<std::thread> threads;
    std::mutex mutex;  // To protect Err's insertions
    std::vector<Triplet> tripletList;

    // Function for each thread to process a range of rows
    auto worker = [&](int start_row, int end_row) {
        std::vector<Triplet> local_tripletList;
        for (int row = start_row; row < end_row; ++row) {
            for (SpMat::InnerIterator it(X, row); it; ++it) {
                int col = it.col();
                double x_val = it.value();
                double as_val = AS(row, col);
                double err = x_val - as_val;
                if (err != 0.0) {  // Only store non-zero errors
                    local_tripletList.emplace_back(row, col, err);
                }
            }
        }
        // Lock and append to the main triplet list
        std::lock_guard<std::mutex> lock(mutex);
        tripletList.insert(tripletList.end(), local_tripletList.begin(), local_tripletList.end());
    };

    // Determine workload for each thread
    int rows_per_thread = n_rows / numCPU;
    int remaining = n_rows % numCPU;
    int current = 0;

    for (int t = 0; t < numCPU; ++t) {
        int start_row = current;
        int end_row = start_row + rows_per_thread + (t < remaining ? 1 : 0);
        current = end_row;
        threads.emplace_back(worker, start_row, end_row);
    }

    // Join threads
    for (auto& th : threads) {
        th.join();
    }

    // Set the Err matrix from triplet list
    Err.setFromTriplets(tripletList.begin(), tripletList.end());

    // Convert Err to CSR components
    std::vector<double> Err_data;
    std::vector<int> Err_indices;
    std::vector<int> Err_indptr;
    eigen_to_csr(Err, Err_data, Err_indices, Err_indptr);

    // Create Python arrays
    py::array_t<double> Err_data_py(Err_data.size(), Err_data.data());
    py::array_t<int> Err_indices_py(Err_indices.size(), Err_indices.data());
    py::array_t<int> Err_indptr_py(Err_indptr.size(), Err_indptr.data());

    // Create a dictionary to hold CSR components
    py::dict result;
    result["data"] = Err_data_py;
    result["indices"] = Err_indices_py;
    result["indptr"] = Err_indptr_py;
    result["shape"] = py::make_tuple(n_rows, n_cols);

    return result;
}

PYBIND11_MODULE(errpca_pt, m) {
    m.doc() = "errpca_pt: Compute sparse matrix of reconstruction errors";

    m.def("errpca_pt", &errpca_pt,
          py::arg("X_data"),
          py::arg("X_indices"),
          py::arg("X_indptr"),
          py::arg("A"),
          py::arg("S"),
          py::arg("numCPU") = 1,
          "Compute sparse matrix of reconstruction errors (X - A*S) with sparsity of X.");
}
