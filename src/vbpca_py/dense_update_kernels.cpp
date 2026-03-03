#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Dense>

#include <algorithm>
#include <stdexcept>
#include <thread>

namespace py = pybind11;

namespace {

constexpr double EPS_JITTER = 1e-15;

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
    const Eigen::MatrixXd &x_data,
    const Eigen::MatrixXd &mask,
    const Eigen::MatrixXd &loadings,
    const py::object &loading_covariances_obj,
    double noise_var,
    bool return_covariances,
    int num_cpu
) {
    const int n_features = static_cast<int>(x_data.rows());
    const int n_samples = static_cast<int>(x_data.cols());
    const int n_components = static_cast<int>(loadings.cols());

    if (static_cast<int>(loadings.rows()) != n_features) {
        throw std::invalid_argument("loadings row count must match x_data rows.");
    }
    if (
        static_cast<int>(mask.rows()) != n_features ||
        static_cast<int>(mask.cols()) != n_samples
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
    Eigen::MatrixXd score_covariances;
    if (return_covariances) {
        score_covariances = Eigen::MatrixXd::Zero(n_samples, n_components * n_components);
    }

    const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(n_components, n_components);

    const int threads_requested = num_cpu > 0 ? num_cpu : static_cast<int>(std::thread::hardware_concurrency());
    const int threads = std::max(1, threads_requested);
    const int actual_threads = std::max(1, std::min(threads, n_samples));

    auto worker = [&](int start, int end) {
        for (int j = start; j < end; ++j) {
            Eigen::MatrixXd psi = noise_var * identity;
            Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n_components);

            for (int i = 0; i < n_features; ++i) {
                if (mask(i, j) <= 0.0) {
                    continue;
                }
                const Eigen::VectorXd a = loadings.row(i).transpose();
                psi.noalias() += a * a.transpose();
                rhs.noalias() += a * x_data(i, j);

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
                        score_covariances(j, r * n_components + c) = sv(r, c);
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
        py::array_t<double> cov_out({n_samples, n_components, n_components});
        auto *cov_ptr = cov_out.mutable_data();
        for (int j = 0; j < n_samples; ++j) {
            for (int r = 0; r < n_components; ++r) {
                for (int c = 0; c < n_components; ++c) {
                    cov_ptr[
                        static_cast<std::size_t>(j) * static_cast<std::size_t>(n_components) *
                            static_cast<std::size_t>(n_components) +
                        static_cast<std::size_t>(r) * static_cast<std::size_t>(n_components) +
                        static_cast<std::size_t>(c)
                    ] = score_covariances(j, r * n_components + c);
                }
            }
        }
        out["score_covariances"] = cov_out;
    }

    return out;
}

py::dict loadings_update_dense_masked_nopattern(
    const Eigen::MatrixXd &x_data,
    const Eigen::MatrixXd &mask,
    const Eigen::MatrixXd &scores,
    const py::object &score_covariances_obj,
    const Eigen::MatrixXd &prior_prec,
    double noise_var,
    bool return_covariances,
    int num_cpu
) {
    const int n_features = static_cast<int>(x_data.rows());
    const int n_samples = static_cast<int>(x_data.cols());
    const int n_components = static_cast<int>(scores.rows());

    if (static_cast<int>(scores.cols()) != n_samples) {
        throw std::invalid_argument("scores column count must match x_data columns.");
    }
    if (
        static_cast<int>(mask.rows()) != n_features ||
        static_cast<int>(mask.cols()) != n_samples
    ) {
        throw std::invalid_argument("mask shape must match x_data shape.");
    }
    if (
        static_cast<int>(prior_prec.rows()) != n_components ||
        static_cast<int>(prior_prec.cols()) != n_components
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
    Eigen::MatrixXd loading_covariances;
    if (return_covariances) {
        loading_covariances = Eigen::MatrixXd::Zero(n_features, n_components * n_components);
    }

    const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(n_components, n_components);

    const int threads_requested = num_cpu > 0 ? num_cpu : static_cast<int>(std::thread::hardware_concurrency());
    const int threads = std::max(1, threads_requested);
    const int actual_threads = std::max(1, std::min(threads, n_features));

    auto worker = [&](int start, int end) {
        for (int i = start; i < end; ++i) {
            Eigen::MatrixXd phi = prior_prec;
            Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n_components);

            for (int j = 0; j < n_samples; ++j) {
                if (mask(i, j) <= 0.0) {
                    continue;
                }
                const Eigen::VectorXd s = scores.col(j);
                phi.noalias() += s * s.transpose();
                rhs.noalias() += s * x_data(i, j);

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
                        loading_covariances(i, r * n_components + c) = av(r, c);
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
        py::array_t<double> cov_out({n_features, n_components, n_components});
        auto *cov_ptr = cov_out.mutable_data();
        for (int i = 0; i < n_features; ++i) {
            for (int r = 0; r < n_components; ++r) {
                for (int c = 0; c < n_components; ++c) {
                    cov_ptr[
                        static_cast<std::size_t>(i) * static_cast<std::size_t>(n_components) *
                            static_cast<std::size_t>(n_components) +
                        static_cast<std::size_t>(r) * static_cast<std::size_t>(n_components) +
                        static_cast<std::size_t>(c)
                    ] = loading_covariances(i, r * n_components + c);
                }
            }
        }
        out["loading_covariances"] = cov_out;
    }

    return out;
}

}  // namespace

PYBIND11_MODULE(dense_update_kernels, m) {
    m.doc() = "Dense fast-path update kernels for fully observed VB-PCA updates.";

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
