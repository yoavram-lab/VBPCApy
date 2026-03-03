#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace py = pybind11;

namespace {

constexpr int AUTO_MIN_ITEMS = 64;
constexpr int AUTO_ITEMS_PER_THREAD = 32;
constexpr int AUTO_SMALL_ITEMS_CAP = 256;

int resolve_num_threads(int requested, int n_items) {
    if (requested > 0) {
        return std::max(1, std::min(requested, n_items));
    }

    const char *env_threads = std::getenv("VBPCA_NUM_THREADS");
    if (env_threads != nullptr) {
        try {
            const int parsed = std::stoi(env_threads);
            if (parsed > 0) {
                return std::max(1, std::min(parsed, n_items));
            }
        } catch (...) {
            // Ignore malformed env value and fall back.
        }
    }

    if (n_items < AUTO_MIN_ITEMS) {
        return 1;
    }

    if (n_items < AUTO_SMALL_ITEMS_CAP) {
        return 1;
    }

    const unsigned int hw = std::thread::hardware_concurrency();
    int threads = hw > 0 ? static_cast<int>(hw) : 1;
    threads = std::max(1, std::min(threads, n_items));
    const int limit_by_items = std::max(1, n_items / AUTO_ITEMS_PER_THREAD);
    threads = std::min(threads, limit_by_items);
    return std::max(1, threads);
}

py::array_t<double> congruence_transform_stack(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &cov_stack,
    const Eigen::MatrixXd &left,
    const Eigen::MatrixXd &right,
    int num_cpu
) {
    const auto cov_buf = cov_stack.request();
    if (cov_buf.ndim != 3) {
        throw std::invalid_argument("cov_stack must be a 3-D array with shape (n_items, k, k).");
    }

    const int n_items = static_cast<int>(cov_buf.shape[0]);
    const int k = static_cast<int>(cov_buf.shape[1]);
    const int k2 = static_cast<int>(cov_buf.shape[2]);
    if (k != k2) {
        throw std::invalid_argument("cov_stack trailing dimensions must be square (k, k).");
    }
    if (n_items < 0 || k < 0) {
        throw std::invalid_argument("cov_stack dimensions must be non-negative.");
    }

    if (left.rows() != k || left.cols() != k) {
        throw std::invalid_argument("left must have shape (k, k) matching cov_stack.");
    }
    if (right.rows() != k || right.cols() != k) {
        throw std::invalid_argument("right must have shape (k, k) matching cov_stack.");
    }

    py::array_t<double> out({n_items, k, k});

    if (n_items == 0 || k == 0) {
        return out;
    }

    const auto *in_ptr = static_cast<const double *>(cov_buf.ptr);
    auto *out_ptr = out.mutable_data();

    const std::size_t mat_elems = static_cast<std::size_t>(k) * static_cast<std::size_t>(k);
    // Respect the incoming numpy strides so we handle both C- and F-contiguous
    // cov_stack inputs without copying or misinterpreting memory layout.
    const std::size_t stride_item = static_cast<std::size_t>(cov_buf.strides[0]) / sizeof(double);
    const std::ptrdiff_t stride_row = cov_buf.strides[1] / static_cast<std::ptrdiff_t>(sizeof(double));
    const std::ptrdiff_t stride_col = cov_buf.strides[2] / static_cast<std::ptrdiff_t>(sizeof(double));
    const Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> eigen_stride(stride_row, stride_col);

    // Use a simple single-threaded loop for correctness and determinism; the
    // matrices are small (k is component count), so threading adds overhead and
    // risks subtle aliasing issues.
    for (int i = 0; i < n_items; ++i) {
        const auto *cov_base = in_ptr + static_cast<std::size_t>(i) * stride_item;

        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> cov_i(
            cov_base, k, k, eigen_stride);

        Eigen::MatrixXd transformed = left * cov_i * right;

        const std::size_t base_out = static_cast<std::size_t>(i) * mat_elems;
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> out_i(
            out_ptr + base_out, k, k);
        out_i = transformed;
    }

    return out;
}

py::dict weighted_cov_eigh_psd(
    const Eigen::MatrixXd &base,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &cov_stack,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &weights,
    double normalizer
) {
    const int k = static_cast<int>(base.rows());
    if (base.cols() != k) {
        throw std::invalid_argument("base must be square with shape (k, k).");
    }
    if (!(normalizer > 0.0)) {
        throw std::invalid_argument("normalizer must be positive.");
    }

    const auto cov_buf = cov_stack.request();
    const auto w_buf = weights.request();

    if (cov_buf.ndim != 3) {
        throw std::invalid_argument("cov_stack must be 3-D with shape (n, k, k).");
    }
    if (w_buf.ndim != 1) {
        throw std::invalid_argument("weights must be 1-D with shape (n,).");
    }

    const int n_items = static_cast<int>(cov_buf.shape[0]);
    const int cov_k1 = static_cast<int>(cov_buf.shape[1]);
    const int cov_k2 = static_cast<int>(cov_buf.shape[2]);
    if (cov_k1 != k || cov_k2 != k) {
        throw std::invalid_argument("cov_stack trailing dimensions must match base shape.");
    }
    if (static_cast<int>(w_buf.shape[0]) != n_items) {
        throw std::invalid_argument("weights length must match cov_stack size.");
    }

    Eigen::MatrixXd cov = base;
    if (n_items > 0) {
        const auto *stack_ptr = static_cast<const double *>(cov_buf.ptr);
        const auto *weights_ptr = static_cast<const double *>(w_buf.ptr);
        const std::size_t mat_elems =
            static_cast<std::size_t>(k) * static_cast<std::size_t>(k);
        for (int i = 0; i < n_items; ++i) {
            const double w = weights_ptr[i];
            if (w == 0.0) {
                continue;
            }
            const std::size_t base_idx = static_cast<std::size_t>(i) * mat_elems;
            Eigen::Map<
                const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            > cov_i(stack_ptr + base_idx, k, k);
            cov.noalias() += w * cov_i;
        }
    }

    cov /= normalizer;
    cov = 0.5 * (cov + cov.transpose());

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(cov);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Eigen decomposition failed in rotate_update_kernels.");
    }

    Eigen::VectorXd eigvals = solver.eigenvalues();
    eigvals = eigvals.cwiseMax(0.0);

    py::dict out;
    out["cov"] = cov;
    out["eigvals"] = eigvals;
    out["eigvecs"] = solver.eigenvectors();
    return out;
}

}  // namespace

PYBIND11_MODULE(rotate_update_kernels, m) {
    m.doc() = "C++ kernels for rotate_to_pca covariance transforms.";

    m.def(
        "congruence_transform_stack",
        &congruence_transform_stack,
        py::arg("cov_stack"),
        py::arg("left"),
        py::arg("right"),
        py::arg("num_cpu") = 0,
        "Apply batched congruence-like transforms to covariance stack.\n\n"
        "For each item i, computes: out[i] = left @ cov_stack[i] @ right."
    );

    m.def(
        "weighted_cov_eigh_psd",
        &weighted_cov_eigh_psd,
        py::arg("base"),
        py::arg("cov_stack"),
        py::arg("weights"),
        py::arg("normalizer"),
        "Build weighted covariance matrix and run symmetric PSD eigendecomposition.\n\n"
        "Computes: cov = (base + sum_i weights[i] * cov_stack[i]) / normalizer,\n"
        "then symmetrizes and returns (cov, eigvals, eigvecs)."
    );
}
