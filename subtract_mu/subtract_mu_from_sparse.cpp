// SUBTRACT_MU.cpp
//
// X = SUBTRACT_MU(X, Mu) subtracts bias term Mu from sparse data
// matrix X.
//
// Equivalent Matlab code:
//   M = spones(X);
//   X = X - repmat(Mu,1,size(X,2)).*M;
//
// This software is provided "as is", without warranty of any kind.
// Alexander Ilin, Tapani Raiko

#include <math.h>
#include <string.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <tuple>
#include <vector>
#include <stdexcept>

namespace py = pybind11;

#define EPS 1e-15

// Function to subtract Mu from sparse matrix X based on row indices
py::array_t<double> subtract_mu_from_sparse(py::array_t<double> mxX_data_array,
                                py::array_t<int> mxX_indices_array,
                                py::array_t<int> mxX_indptr_array,
                                py::tuple mxX_shape,
                                py::array_t<double> mxMu_array)
{
    // Extract data from inputs
    auto X_data = mxX_data_array.unchecked<1>();          // Data array
    auto X_indices = mxX_indices_array.unchecked<1>();    // Column indices array
    auto X_indptr = mxX_indptr_array.unchecked<1>();      // Indptr array
    int n1 = mxX_shape[0].cast<int>();                    // Number of rows
    int n2 = mxX_shape[1].cast<int>();                    // Number of columns
    size_t nzmax = X_indptr[n2];                          // Number of non-zero elements
    auto Mu = mxMu_array.unchecked<1>();                  // Mu array

    // Check if Mu has enough elements
    if (Mu.size() < n1) {
        throw std::invalid_argument("SUBTRACT_MU: Mu vector length does not match the number of rows in X.");
    }

    // Initialize numCPU as in original code
    int numCPU = 1;

    // Generate row_indices based on indptr
    std::vector<int> row_indices(nzmax);
    for(int row = 0; row < n1; row++) {
        for(int idx = X_indptr[row]; idx < X_indptr[row+1]; idx++) {
            row_indices[idx] = row;
        }
    }

    // Prepare output data array
    std::vector<double> Xout(nzmax);

    // Perform the subtraction based on row indices
    if (numCPU ==1)
    {
        for (size_t r = 0; r < nzmax; r++)
        {
            Xout[r] = X_data[r] - Mu[row_indices[r]];
            if (Xout[r] == 0.0)
            {
                Xout[r] = EPS; // Replace exact zeros with EPS
            }
        }
    }
    // Placeholder for multi-threading (numCPU > 1)

    // Create output data array
    py::array_t<double> Xout_array(nzmax);
    auto Xout_mutable = Xout_array.mutable_unchecked<1>();

    for (size_t r = 0; r < nzmax; r++)
    {
        Xout_mutable(r) = Xout[r];
    }

    // Return the modified data array
    return Xout_array;
}

// Binding code
PYBIND11_MODULE(subtract_mu_from_sparse, m)
{
    m.doc() = "Subtract Mu from sparse matrix X using C++ and pybind11";
    m.def("subtract_mu_from_sparse", &subtract_mu_from_sparse, "Subtract Mu from sparse matrix X",
          py::arg("data"), py::arg("indices"), py::arg("indptr"), py::arg("shape"), py::arg("Mu"));
}
