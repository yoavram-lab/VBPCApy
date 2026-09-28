#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>


#include "thread_limits.h"
namespace py = pybind11;

namespace {
using DoubleArray = py::array_t<double, py::array::forcecast>;

struct MatrixView {
    const double *data;
    int rows;
    int cols;
    py::ssize_t row_stride;
    py::ssize_t col_stride;

    double operator()(int row, int col) const {
        return data[
            static_cast<py::ssize_t>(row) * row_stride +
            static_cast<py::ssize_t>(col) * col_stride
        ];
    }
};

MatrixView matrix_view(const DoubleArray &array, const char *name) {
    const auto info = array.request();
    if (info.ndim != 2) {
        throw std::invalid_argument(std::string(name) + " must be a 2-D array.");
    }
    const bool c_contiguous = (array.flags() & py::array::c_style) != 0;
    const bool f_contiguous = (array.flags() & py::array::f_style) != 0;
    if (!c_contiguous && !f_contiguous) {
        throw std::invalid_argument(
            std::string(name) + " must be C- or Fortran-contiguous."
        );
    }
    if (
        info.strides[0] % static_cast<py::ssize_t>(sizeof(double)) != 0 ||
        info.strides[1] % static_cast<py::ssize_t>(sizeof(double)) != 0
    ) {
        throw std::invalid_argument(std::string(name) + " has invalid strides.");
    }
    return {
        static_cast<const double *>(info.ptr),
        static_cast<int>(info.shape[0]),
        static_cast<int>(info.shape[1]),
        info.strides[0] / static_cast<py::ssize_t>(sizeof(double)),
        info.strides[1] / static_cast<py::ssize_t>(sizeof(double)),
    };
}

enum class MaskType { boolean, uint8, float64 };

struct MaskView {
    const char *data;
    int rows;
    int cols;
    py::ssize_t row_stride;
    py::ssize_t col_stride;
    MaskType type;

    bool observed(int row, int col) const {
        const char *value = data + static_cast<py::ssize_t>(row) * row_stride +
                            static_cast<py::ssize_t>(col) * col_stride;
        if (type == MaskType::boolean) {
            return *reinterpret_cast<const bool *>(value);
        }
        if (type == MaskType::uint8) {
            return *reinterpret_cast<const std::uint8_t *>(value) != 0;
        }
        return *reinterpret_cast<const double *>(value) > 0.0;
    }
};

MaskView mask_view(const py::array &array) {
    const auto info = array.request();
    if (info.ndim != 2) {
        throw std::invalid_argument("mask must be a 2-D array.");
    }
    const bool c_contiguous = (array.flags() & py::array::c_style) != 0;
    const bool f_contiguous = (array.flags() & py::array::f_style) != 0;
    if (!c_contiguous && !f_contiguous) {
        throw std::invalid_argument("mask must be C- or Fortran-contiguous.");
    }

    MaskType type;
    if (array.dtype().is(py::dtype::of<bool>())) {
        type = MaskType::boolean;
    } else if (array.dtype().is(py::dtype::of<std::uint8_t>())) {
        type = MaskType::uint8;
    } else if (array.dtype().is(py::dtype::of<double>())) {
        type = MaskType::float64;
    } else {
        throw std::invalid_argument("mask dtype must be bool, uint8, or float64.");
    }
    return {
        static_cast<const char *>(info.ptr),
        static_cast<int>(info.shape[0]),
        static_cast<int>(info.shape[1]),
        info.strides[0],
        info.strides[1],
        type,
    };
}

const char *mask_type_name(MaskType type) {
    if (type == MaskType::boolean) {
        return "bool";
    }
    if (type == MaskType::uint8) {
        return "uint8";
    }
    return "float64";
}


constexpr double EPS_JITTER = 1e-15;
py::dict inspect_dense_input_views(
    const DoubleArray &x_data_array,
    const py::array &mask_array
) {
    const MatrixView x_data = matrix_view(x_data_array, "x_data");
    const MaskView mask = mask_view(mask_array);
    if (x_data.rows != mask.rows || x_data.cols != mask.cols) {
        throw std::invalid_argument("mask shape must match x_data shape.");
    }

    py::dict out;
    out["x_pointer"] = py::int_(
        reinterpret_cast<std::uintptr_t>(x_data.data)
    );
    out["mask_pointer"] = py::int_(
        reinterpret_cast<std::uintptr_t>(mask.data)
    );
    out["x_strides_elements"] = py::make_tuple(
        x_data.row_stride,
        x_data.col_stride
    );
    out["mask_strides_bytes"] = py::make_tuple(
        mask.row_stride,
        mask.col_stride
    );
    out["mask_dtype"] = mask_type_name(mask.type);
    out["x_c_contiguous"] =
        (x_data_array.flags() & py::array::c_style) != 0;
    out["x_f_contiguous"] =
        (x_data_array.flags() & py::array::f_style) != 0;
    out["mask_c_contiguous"] =
        (mask_array.flags() & py::array::c_style) != 0;
    out["mask_f_contiguous"] =
        (mask_array.flags() & py::array::f_style) != 0;
    return out;
}


Eigen::LLT<Eigen::MatrixXd> stable_llt(Eigen::MatrixXd mat) {
    // Symmetrize to avoid tiny asymmetries from prior operations.
    mat = 0.5 * (mat + mat.transpose());
    Eigen::LLT<Eigen::MatrixXd> llt;
    llt.compute(mat);
    if (llt.info() == Eigen::Success) {
        return llt;
    }

    const Eigen::Index n = mat.rows();
    mat.diagonal().array() += EPS_JITTER;
    llt.compute(mat);
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error("Cholesky factorization failed in dense update kernel.");
    }
    if (mat.cols() != n) {
        throw std::runtime_error("Invalid matrix dimensions for Cholesky.");
    }
    return llt;
}

py::dict score_update_dense_no_av(
    const Eigen::MatrixXd &x_data,
    const Eigen::MatrixXd &loadings,
    double noise_var,
    bool return_covariance
) {
    const int n_features = static_cast<int>(x_data.rows());
    const int n_samples = static_cast<int>(x_data.cols());
    const int n_components = static_cast<int>(loadings.cols());

    if (static_cast<int>(loadings.rows()) != n_features) {
        throw std::invalid_argument("loadings row count must match x_data rows.");
    }

    const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(n_components, n_components);
    Eigen::MatrixXd psi = loadings.transpose() * loadings + noise_var * identity;
    const Eigen::LLT<Eigen::MatrixXd> llt = stable_llt(std::move(psi));

    const Eigen::MatrixXd rhs = loadings.transpose() * x_data;
    const Eigen::MatrixXd scores = llt.solve(rhs);

    py::dict out;
    out["scores"] = scores;

    if (return_covariance) {
        const Eigen::MatrixXd score_cov = noise_var * llt.solve(identity);
        out["score_covariance"] = score_cov;
    }

    return out;
}

py::dict loadings_update_dense_no_sv(
    const Eigen::MatrixXd &x_data,
    const Eigen::MatrixXd &scores,
    const Eigen::MatrixXd &prior_prec,
    double noise_var,
    bool return_covariance
) {
    const int n_features = static_cast<int>(x_data.rows());
    const int n_samples = static_cast<int>(x_data.cols());
    const int n_components = static_cast<int>(scores.rows());

    if (static_cast<int>(scores.cols()) != n_samples) {
        throw std::invalid_argument("scores column count must match x_data columns.");
    }
    if (
        static_cast<int>(prior_prec.rows()) != n_components ||
        static_cast<int>(prior_prec.cols()) != n_components
    ) {
        throw std::invalid_argument("prior_prec must be square with size n_components.");
    }

    Eigen::MatrixXd phi = scores * scores.transpose() + prior_prec;
    const Eigen::LLT<Eigen::MatrixXd> llt = stable_llt(std::move(phi));

    const Eigen::MatrixXd rhs = scores * x_data.transpose();
    const Eigen::MatrixXd loadings = llt.solve(rhs).transpose();

    py::dict out;
    out["loadings"] = loadings;

    if (return_covariance) {
        const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(n_components, n_components);
        const Eigen::MatrixXd loading_cov = noise_var * llt.solve(identity);
        out["loading_covariance"] = loading_cov;
    }

    return out;
}

py::dict score_update_dense_masked_nopattern(
    const DoubleArray &x_data_array,
    const py::array &mask_array,
    const DoubleArray &loadings_array,
    const py::object &loading_covariances_obj,
    double noise_var,
    bool return_covariances,
    int num_cpu
) {
    const MatrixView x_data = matrix_view(x_data_array, "x_data");
    const MaskView mask = mask_view(mask_array);
    const MatrixView loadings = matrix_view(loadings_array, "loadings");
    const int n_features = x_data.rows;
    const int n_samples = x_data.cols;
    const int n_components = loadings.cols;

    if (loadings.rows != n_features) {
        throw std::invalid_argument("loadings row count must match x_data rows.");
    }
    if (
        mask.rows != n_features ||
        mask.cols != n_samples
    ) {
        throw std::invalid_argument("mask shape must match x_data shape.");
    }

    py::array_t<double, py::array::c_style | py::array::forcecast> av_arr;
    const double *av_ptr = nullptr;
    if (!loading_covariances_obj.is_none()) {
        av_arr = py::array_t<double, py::array::c_style | py::array::forcecast>(loading_covariances_obj);
        auto av_buf = av_arr.request();
        if (av_buf.ndim != 3) {
            throw std::invalid_argument("loading_covariances must be 3-D.");
        }
        if (
            av_buf.shape[0] != n_features ||
            av_buf.shape[1] != n_components ||
            av_buf.shape[2] != n_components
        ) {
            throw std::invalid_argument("loading_covariances shape mismatch.");
        }
        av_ptr = static_cast<const double *>(av_buf.ptr);
    }

    Eigen::MatrixXd scores = Eigen::MatrixXd::Zero(n_components, n_samples);
    py::array_t<double> score_covariances;
    double *score_covariances_ptr = nullptr;
    if (return_covariances) {
        score_covariances = py::array_t<double>({n_samples, n_components, n_components});
        score_covariances_ptr = score_covariances.mutable_data();
    }

    const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(n_components, n_components);

    const int actual_threads = vbpca_threads::resolve_thread_count(num_cpu, n_samples);

    auto worker = [&](int start, int end) {
        Eigen::VectorXd component(n_components);
        Eigen::VectorXd rhs(n_components);
        for (int j = start; j < end; ++j) {
            Eigen::MatrixXd psi = noise_var * identity;
            rhs.setZero();

            for (int i = 0; i < n_features; ++i) {
                if (!mask.observed(i, j)) {
                    continue;
                }
                for (int r = 0; r < n_components; ++r) {
                    component(r) = loadings(i, r);
                }
                psi.noalias() += component * component.transpose();
                rhs.noalias() += component * x_data(i, j);

                if (av_ptr != nullptr) {
                    const std::size_t base =
                        static_cast<std::size_t>(i) * static_cast<std::size_t>(n_components) *
                        static_cast<std::size_t>(n_components);
                    for (int r = 0; r < n_components; ++r) {
                        for (int c = 0; c < n_components; ++c) {
                            psi(r, c) += av_ptr[
                                base + static_cast<std::size_t>(r) * static_cast<std::size_t>(n_components) +
                                static_cast<std::size_t>(c)
                            ];
                        }
                    }
                }
            }

            const Eigen::LLT<Eigen::MatrixXd> llt = stable_llt(std::move(psi));
            scores.col(j) = llt.solve(rhs);

            if (return_covariances) {
                const Eigen::MatrixXd sv = noise_var * llt.solve(identity);
                for (int r = 0; r < n_components; ++r) {
                    for (int c = 0; c < n_components; ++c) {
                        score_covariances_ptr[
                            static_cast<std::size_t>(j) * static_cast<std::size_t>(n_components) *
                                static_cast<std::size_t>(n_components) +
                            static_cast<std::size_t>(r) * static_cast<std::size_t>(n_components) +
                            static_cast<std::size_t>(c)
                        ] = sv(r, c);
                    }
                }
            }
        }
    };

    {
        py::gil_scoped_release release;

        if (actual_threads <= 1) {
            worker(0, n_samples);
        } else {
            std::vector<std::thread> pool;
            pool.reserve(static_cast<std::size_t>(actual_threads));
            const int rows_per_thread = n_samples / actual_threads;
            const int remainder = n_samples % actual_threads;
            int current = 0;
            for (int t = 0; t < actual_threads; ++t) {
                const int extra = (t < remainder) ? 1 : 0;
                const int start = current;
                const int end = start + rows_per_thread + extra;
                current = end;
                pool.emplace_back(worker, start, end);
            }
            for (auto &th : pool) {
                th.join();
            }
        }
    }

    py::dict out;
    out["scores"] = scores;

    if (return_covariances) {
        out["score_covariances"] = score_covariances;
    }

    return out;
}

py::dict loadings_update_dense_masked_nopattern(
    const DoubleArray &x_data_array,
    const py::array &mask_array,
    const DoubleArray &scores_array,
    const py::object &score_covariances_obj,
    const DoubleArray &prior_prec_array,
    double noise_var,
    bool return_covariances,
    int num_cpu
) {
    const MatrixView x_data = matrix_view(x_data_array, "x_data");
    const MaskView mask = mask_view(mask_array);
    const MatrixView scores = matrix_view(scores_array, "scores");
    const MatrixView prior_prec = matrix_view(prior_prec_array, "prior_prec");
    const int n_features = x_data.rows;
    const int n_samples = x_data.cols;
    const int n_components = scores.rows;

    if (scores.cols != n_samples) {
        throw std::invalid_argument("scores column count must match x_data columns.");
    }
    if (
        mask.rows != n_features ||
        mask.cols != n_samples
    ) {
        throw std::invalid_argument("mask shape must match x_data shape.");
    }
    if (
        prior_prec.rows != n_components ||
        prior_prec.cols != n_components
    ) {
        throw std::invalid_argument("prior_prec must be square with size n_components.");
    }

    py::array_t<double, py::array::c_style | py::array::forcecast> sv_arr;
    const double *sv_ptr = nullptr;
    if (!score_covariances_obj.is_none()) {
        sv_arr = py::array_t<double, py::array::c_style | py::array::forcecast>(score_covariances_obj);
        auto sv_buf = sv_arr.request();
        if (sv_buf.ndim != 3) {
            throw std::invalid_argument("score_covariances must be 3-D.");
        }
        if (
            sv_buf.shape[0] != n_samples ||
            sv_buf.shape[1] != n_components ||
            sv_buf.shape[2] != n_components
        ) {
            throw std::invalid_argument("score_covariances shape mismatch.");
        }
        sv_ptr = static_cast<const double *>(sv_buf.ptr);
    }

    Eigen::MatrixXd loadings = Eigen::MatrixXd::Zero(n_features, n_components);
    py::array_t<double> loading_covariances;
    double *loading_covariances_ptr = nullptr;
    if (return_covariances) {
        loading_covariances = py::array_t<double>({n_features, n_components, n_components});
        loading_covariances_ptr = loading_covariances.mutable_data();
    }

    const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(n_components, n_components);

    const int actual_threads = vbpca_threads::resolve_thread_count(num_cpu, n_features);

    auto worker = [&](int start, int end) {
        Eigen::VectorXd component(n_components);
        Eigen::VectorXd rhs(n_components);
        for (int i = start; i < end; ++i) {
            Eigen::MatrixXd phi(n_components, n_components);
            for (int r = 0; r < n_components; ++r) {
                for (int c = 0; c < n_components; ++c) {
                    phi(r, c) = prior_prec(r, c);
                }
            }
            rhs.setZero();

            for (int j = 0; j < n_samples; ++j) {
                if (!mask.observed(i, j)) {
                    continue;
                }
                for (int r = 0; r < n_components; ++r) {
                    component(r) = scores(r, j);
                }
                phi.noalias() += component * component.transpose();
                rhs.noalias() += component * x_data(i, j);

                if (sv_ptr != nullptr) {
                    const std::size_t base =
                        static_cast<std::size_t>(j) * static_cast<std::size_t>(n_components) *
                        static_cast<std::size_t>(n_components);
                    for (int r = 0; r < n_components; ++r) {
                        for (int c = 0; c < n_components; ++c) {
                            phi(r, c) += sv_ptr[
                                base + static_cast<std::size_t>(r) * static_cast<std::size_t>(n_components) +
                                static_cast<std::size_t>(c)
                            ];
                        }
                    }
                }
            }

            const Eigen::LLT<Eigen::MatrixXd> llt = stable_llt(std::move(phi));
            loadings.row(i) = llt.solve(rhs).transpose();

            if (return_covariances) {
                const Eigen::MatrixXd av = noise_var * llt.solve(identity);
                for (int r = 0; r < n_components; ++r) {
                    for (int c = 0; c < n_components; ++c) {
                        loading_covariances_ptr[
                            static_cast<std::size_t>(i) * static_cast<std::size_t>(n_components) *
                                static_cast<std::size_t>(n_components) +
                            static_cast<std::size_t>(r) * static_cast<std::size_t>(n_components) +
                            static_cast<std::size_t>(c)
                        ] = av(r, c);
                    }
                }
            }
        }
    };

    {
        py::gil_scoped_release release;

        if (actual_threads <= 1) {
            worker(0, n_features);
        } else {
            std::vector<std::thread> pool;
            pool.reserve(static_cast<std::size_t>(actual_threads));
            const int rows_per_thread = n_features / actual_threads;
            const int remainder = n_features % actual_threads;
            int current = 0;
            for (int t = 0; t < actual_threads; ++t) {
                const int extra = (t < remainder) ? 1 : 0;
                const int start = current;
                const int end = start + rows_per_thread + extra;
                current = end;
                pool.emplace_back(worker, start, end);
            }
            for (auto &th : pool) {
                th.join();
            }
        }
    }

    py::dict out;
    out["loadings"] = loadings;

    if (return_covariances) {
        out["loading_covariances"] = loading_covariances;
    }

    return out;
}

}  // namespace

PYBIND11_MODULE(dense_update_kernels, m) {
    m.doc() = "Dense fast-path update kernels for fully observed VB-PCA updates.";

    m.def(
        "inspect_dense_input_views",
        &inspect_dense_input_views,
        py::arg("x_data"),
        py::arg("mask")
    );

    m.def(
        "score_update_dense_no_av",
        &score_update_dense_no_av,
        py::arg("x_data"),
        py::arg("loadings"),
        py::arg("noise_var"),
        py::arg("return_covariance") = true
    );

    m.def(
        "loadings_update_dense_no_sv",
        &loadings_update_dense_no_sv,
        py::arg("x_data"),
        py::arg("scores"),
        py::arg("prior_prec"),
        py::arg("noise_var"),
        py::arg("return_covariance") = true
    );

    m.def(
        "score_update_dense_masked_nopattern",
        &score_update_dense_masked_nopattern,
        py::arg("x_data"),
        py::arg("mask"),
        py::arg("loadings"),
        py::arg("loading_covariances") = py::none(),
        py::arg("noise_var"),
        py::arg("return_covariances") = true,
        py::arg("num_cpu") = 0
    );

    m.def(
        "loadings_update_dense_masked_nopattern",
        &loadings_update_dense_masked_nopattern,
        py::arg("x_data"),
        py::arg("mask"),
        py::arg("scores"),
        py::arg("score_covariances") = py::none(),
        py::arg("prior_prec"),
        py::arg("noise_var"),
        py::arg("return_covariances") = true,
        py::arg("num_cpu") = 0
    );
}
