#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace py = pybind11;

namespace {

constexpr double EPS_JITTER = 1e-15;
constexpr int AUTO_MIN_ITEMS = 64;
constexpr int AUTO_ITEMS_PER_THREAD = 32;
constexpr long long AUTO_WORK_PER_THREAD = 80000;
constexpr int AUTO_SMALL_ITEMS_CAP = 400;
constexpr long long AUTO_SMALL_WORK_CAP = 120000;

int resolve_num_threads(int requested, int n_items, long long work_units) {
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

    if (n_items < AUTO_SMALL_ITEMS_CAP && work_units < AUTO_SMALL_WORK_CAP) {
        return 1;
    }

    const unsigned int hw_threads = std::thread::hardware_concurrency();
    int threads = hw_threads > 0 ? static_cast<int>(hw_threads) : 1;
    threads = std::max(1, std::min(threads, n_items));

    const int limit_by_items = std::max(1, n_items / AUTO_ITEMS_PER_THREAD);
    const int limit_by_work = std::max(
        1,
        static_cast<int>(work_units / AUTO_WORK_PER_THREAD)
    );
    threads = std::min(threads, limit_by_items);
    threads = std::min(threads, limit_by_work);
    return std::max(1, threads);
}

void validate_csc(const py::buffer_info &data, const py::buffer_info &indices, const py::buffer_info &indptr) {
    if (data.ndim != 1 || indices.ndim != 1 || indptr.ndim != 1) {
        throw std::invalid_argument("CSC arrays must be 1-D.");
    }
    if (data.size != indices.size) {
        throw std::invalid_argument("CSC data and indices must have equal length.");
    }
    if (indptr.size < 1) {
        throw std::invalid_argument("CSC indptr length must be >= 1.");
    }
}

void validate_csr(const py::buffer_info &data, const py::buffer_info &indices, const py::buffer_info &indptr) {
    if (data.ndim != 1 || indices.ndim != 1 || indptr.ndim != 1) {
        throw std::invalid_argument("CSR arrays must be 1-D.");
    }
    if (data.size != indices.size) {
        throw std::invalid_argument("CSR data and indices must have equal length.");
    }
    if (indptr.size < 1) {
        throw std::invalid_argument("CSR indptr length must be >= 1.");
    }
}

Eigen::LLT<Eigen::MatrixXd> stable_llt(Eigen::MatrixXd mat) {
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
        throw std::runtime_error("Cholesky factorization failed in sparse update kernel.");
    }
    if (mat.cols() != n) {
        throw std::runtime_error("Invalid matrix dimensions for Cholesky.");
    }
    return llt;
}

py::dict score_update_sparse_nopattern(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &x_data_arr,
    const py::array_t<int, py::array::c_style | py::array::forcecast> &x_indices_arr,
    const py::array_t<int, py::array::c_style | py::array::forcecast> &x_indptr_arr,
    const Eigen::MatrixXd &loadings,
    const py::object &loading_covariances_obj,
    double noise_var,
    bool return_covariances,
    int num_cpu
) {
    const auto data_buf = x_data_arr.request();
    const auto indices_buf = x_indices_arr.request();
    const auto indptr_buf = x_indptr_arr.request();
    validate_csc(data_buf, indices_buf, indptr_buf);

    const auto *x_data = static_cast<const double *>(data_buf.ptr);
    const auto *x_indices = static_cast<const int *>(indices_buf.ptr);
    const auto *x_indptr = static_cast<const int *>(indptr_buf.ptr);

    const int n_samples = static_cast<int>(indptr_buf.size) - 1;
    const int n_features = static_cast<int>(loadings.rows());
    const int n_components = static_cast<int>(loadings.cols());
    const int nnz = static_cast<int>(data_buf.size);

    if (x_indptr[0] != 0 || x_indptr[n_samples] != nnz) {
        throw std::invalid_argument("Invalid CSC indptr endpoints.");
    }

    py::array_t<double, py::array::c_style | py::array::forcecast> av_arr;
    const double *av_ptr = nullptr;
    if (!loading_covariances_obj.is_none()) {
        av_arr = py::array_t<double, py::array::c_style | py::array::forcecast>(
            loading_covariances_obj
        );
        auto av_buf = av_arr.request();
        if (av_buf.ndim != 3) {
            throw std::invalid_argument("loading_covariances must be a 3-D array.");
        }
        if (av_buf.shape[0] != n_features || av_buf.shape[1] != n_components || av_buf.shape[2] != n_components) {
            throw std::invalid_argument("loading_covariances shape mismatch.");
        }
        av_ptr = static_cast<const double *>(av_buf.ptr);
    }

    py::array_t<double> scores_out({n_components, n_samples});
    auto *scores_ptr = scores_out.mutable_data();

    py::array_t<double> cov_out;
    double *cov_ptr = nullptr;
    if (return_covariances) {
        cov_out = py::array_t<double>({n_samples, n_components, n_components});
        cov_ptr = cov_out.mutable_data();
    }

    const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(n_components, n_components);

    for (int p = 0; p < nnz; ++p) {
        const int i = x_indices[p];
        if (i < 0 || i >= n_features) {
            throw std::invalid_argument("CSC row index out of range.");
        }
    }

    const long long work_units =
        static_cast<long long>(nnz) * static_cast<long long>(std::max(1, n_components));
    const int threads = resolve_num_threads(num_cpu, n_samples, work_units);
    std::exception_ptr worker_error;
    std::mutex error_mutex;

    auto worker = [&](int col_start, int col_end) {
        try {
            for (int j = col_start; j < col_end; ++j) {
                const int start = x_indptr[j];
                const int end = x_indptr[j + 1];

                Eigen::MatrixXd psi = noise_var * identity;
                Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n_components);

                for (int p = start; p < end; ++p) {
                    const int i = x_indices[p];
                    const Eigen::VectorXd a = loadings.row(i).transpose();
                    psi.noalias() += a * a.transpose();
                    rhs.noalias() += a * x_data[p];

                    if (av_ptr != nullptr) {
                        const std::size_t base = static_cast<std::size_t>(i) *
                                                 static_cast<std::size_t>(n_components) *
                                                 static_cast<std::size_t>(n_components);
                        for (int r = 0; r < n_components; ++r) {
                            for (int c = 0; c < n_components; ++c) {
                                psi(r, c) += av_ptr[
                                    base + static_cast<std::size_t>(r) *
                                               static_cast<std::size_t>(n_components) +
                                    static_cast<std::size_t>(c)
                                ];
                            }
                        }
                    }
                }

                const Eigen::LLT<Eigen::MatrixXd> llt = stable_llt(std::move(psi));
                const Eigen::VectorXd score_col = llt.solve(rhs);

                for (int r = 0; r < n_components; ++r) {
                    scores_ptr[
                        static_cast<std::size_t>(r) * static_cast<std::size_t>(n_samples) +
                        static_cast<std::size_t>(j)
                    ] = score_col(r);
                }

                if (return_covariances) {
                    const Eigen::MatrixXd sv = noise_var * llt.solve(identity);
                    const std::size_t base = static_cast<std::size_t>(j) *
                                             static_cast<std::size_t>(n_components) *
                                             static_cast<std::size_t>(n_components);
                    for (int r = 0; r < n_components; ++r) {
                        for (int c = 0; c < n_components; ++c) {
                            cov_ptr[
                                base + static_cast<std::size_t>(r) *
                                           static_cast<std::size_t>(n_components) +
                                static_cast<std::size_t>(c)
                            ] = sv(r, c);
                        }
                    }
                }
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(error_mutex);
            if (!worker_error) {
                worker_error = std::current_exception();
            }
        }
    };

    {
        py::gil_scoped_release release;

        if (threads == 1) {
            worker(0, n_samples);
        } else {
            std::vector<std::thread> pool;
            pool.reserve(static_cast<std::size_t>(threads));

            const int cols_per_thread = n_samples / threads;
            const int remainder = n_samples % threads;

            int current = 0;
            for (int t = 0; t < threads; ++t) {
                const int extra = (t < remainder) ? 1 : 0;
                const int start = current;
                const int end = start + cols_per_thread + extra;
                current = end;
                pool.emplace_back(worker, start, end);
            }

            for (auto &th : pool) {
                th.join();
            }
        }
    }

    if (worker_error) {
        std::rethrow_exception(worker_error);
    }

    py::dict out;
    out["scores"] = scores_out;
    if (return_covariances) {
        out["score_covariances"] = cov_out;
    }
    return out;
}

py::dict loadings_update_sparse_nopattern(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &x_data_arr,
    const py::array_t<int, py::array::c_style | py::array::forcecast> &x_indices_arr,
    const py::array_t<int, py::array::c_style | py::array::forcecast> &x_indptr_arr,
    const Eigen::MatrixXd &scores,
    const py::object &score_covariances_obj,
    const Eigen::MatrixXd &prior_prec,
    double noise_var,
    bool return_covariances,
    int num_cpu
) {
    const auto data_buf = x_data_arr.request();
    const auto indices_buf = x_indices_arr.request();
    const auto indptr_buf = x_indptr_arr.request();
    validate_csr(data_buf, indices_buf, indptr_buf);

    const auto *x_data = static_cast<const double *>(data_buf.ptr);
    const auto *x_indices = static_cast<const int *>(indices_buf.ptr);
    const auto *x_indptr = static_cast<const int *>(indptr_buf.ptr);

    const int n_features = static_cast<int>(indptr_buf.size) - 1;
    const int n_samples = static_cast<int>(scores.cols());
    const int n_components = static_cast<int>(scores.rows());
    const int nnz = static_cast<int>(data_buf.size);

    if (x_indptr[0] != 0 || x_indptr[n_features] != nnz) {
        throw std::invalid_argument("Invalid CSR indptr endpoints.");
    }
    if (prior_prec.rows() != n_components || prior_prec.cols() != n_components) {
        throw std::invalid_argument("prior_prec shape mismatch.");
    }

    py::array_t<double, py::array::c_style | py::array::forcecast> sv_arr;
    const double *sv_ptr = nullptr;
    if (!score_covariances_obj.is_none()) {
        sv_arr = py::array_t<double, py::array::c_style | py::array::forcecast>(
            score_covariances_obj
        );
        auto sv_buf = sv_arr.request();
        if (sv_buf.ndim != 3) {
            throw std::invalid_argument("score_covariances must be a 3-D array.");
        }
        if (sv_buf.shape[0] != n_samples || sv_buf.shape[1] != n_components || sv_buf.shape[2] != n_components) {
            throw std::invalid_argument("score_covariances shape mismatch.");
        }
        sv_ptr = static_cast<const double *>(sv_buf.ptr);
    }

    py::array_t<double> loadings_out({n_features, n_components});
    auto *loadings_ptr = loadings_out.mutable_data();

    py::array_t<double> cov_out;
    double *cov_ptr = nullptr;
    if (return_covariances) {
        cov_out = py::array_t<double>({n_features, n_components, n_components});
        cov_ptr = cov_out.mutable_data();
    }

    const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(n_components, n_components);

    for (int p = 0; p < nnz; ++p) {
        const int j = x_indices[p];
        if (j < 0 || j >= n_samples) {
            throw std::invalid_argument("CSR column index out of range.");
        }
    }

    const long long work_units =
        static_cast<long long>(nnz) * static_cast<long long>(std::max(1, n_components));
    const int threads = resolve_num_threads(num_cpu, n_features, work_units);
    std::exception_ptr worker_error;
    std::mutex error_mutex;

    auto worker = [&](int row_start, int row_end) {
        try {
            for (int i = row_start; i < row_end; ++i) {
                const int start = x_indptr[i];
                const int end = x_indptr[i + 1];

                Eigen::MatrixXd phi = prior_prec;
                Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n_components);

                for (int p = start; p < end; ++p) {
                    const int j = x_indices[p];
                    const Eigen::VectorXd s = scores.col(j);
                    phi.noalias() += s * s.transpose();
                    rhs.noalias() += s * x_data[p];

                    if (sv_ptr != nullptr) {
                        const std::size_t base = static_cast<std::size_t>(j) *
                                                 static_cast<std::size_t>(n_components) *
                                                 static_cast<std::size_t>(n_components);
                        for (int r = 0; r < n_components; ++r) {
                            for (int c = 0; c < n_components; ++c) {
                                phi(r, c) += sv_ptr[
                                    base + static_cast<std::size_t>(r) *
                                               static_cast<std::size_t>(n_components) +
                                    static_cast<std::size_t>(c)
                                ];
                            }
                        }
                    }
                }

                const Eigen::LLT<Eigen::MatrixXd> llt = stable_llt(std::move(phi));
                const Eigen::VectorXd a_row = llt.solve(rhs);

                for (int c = 0; c < n_components; ++c) {
                    loadings_ptr[
                        static_cast<std::size_t>(i) * static_cast<std::size_t>(n_components) +
                        static_cast<std::size_t>(c)
                    ] = a_row(c);
                }

                if (return_covariances) {
                    const Eigen::MatrixXd av = noise_var * llt.solve(identity);
                    const std::size_t base = static_cast<std::size_t>(i) *
                                             static_cast<std::size_t>(n_components) *
                                             static_cast<std::size_t>(n_components);
                    for (int r = 0; r < n_components; ++r) {
                        for (int c = 0; c < n_components; ++c) {
                            cov_ptr[
                                base + static_cast<std::size_t>(r) *
                                           static_cast<std::size_t>(n_components) +
                                static_cast<std::size_t>(c)
                            ] = av(r, c);
                        }
                    }
                }
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(error_mutex);
            if (!worker_error) {
                worker_error = std::current_exception();
            }
        }
    };

    {
        py::gil_scoped_release release;

        if (threads == 1) {
            worker(0, n_features);
        } else {
            std::vector<std::thread> pool;
            pool.reserve(static_cast<std::size_t>(threads));

            const int rows_per_thread = n_features / threads;
            const int remainder = n_features % threads;

            int current = 0;
            for (int t = 0; t < threads; ++t) {
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

    if (worker_error) {
        std::rethrow_exception(worker_error);
    }

    py::dict out;
    out["loadings"] = loadings_out;
    if (return_covariances) {
        out["loading_covariances"] = cov_out;
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(sparse_update_kernels, m) {
    m.doc() = "Sparse score/loadings update kernels for pattern-free VB-PCA updates.";

    m.def(
        "score_update_sparse_nopattern",
        &score_update_sparse_nopattern,
        py::arg("x_data"),
        py::arg("x_indices"),
        py::arg("x_indptr"),
        py::arg("loadings"),
        py::arg("loading_covariances") = py::none(),
        py::arg("noise_var"),
        py::arg("return_covariances") = true,
        py::arg("num_cpu") = 0
    );

    m.def(
        "loadings_update_sparse_nopattern",
        &loadings_update_sparse_nopattern,
        py::arg("x_data"),
        py::arg("x_indices"),
        py::arg("x_indptr"),
        py::arg("scores"),
        py::arg("score_covariances") = py::none(),
        py::arg("prior_prec"),
        py::arg("noise_var"),
        py::arg("return_covariances") = true,
        py::arg("num_cpu") = 0
    );
}
