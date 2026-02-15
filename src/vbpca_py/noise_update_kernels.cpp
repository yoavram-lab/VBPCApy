#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace py = pybind11;

namespace {

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
            // Ignore malformed env and fall back.
        }
    }

    const unsigned int hw = std::thread::hardware_concurrency();
    const int fallback = hw > 0 ? static_cast<int>(hw) : 1;
    return std::max(1, std::min(fallback, n_items));
}

std::size_t idx3(int a, int b, int c, int dim_b, int dim_c) {
    return static_cast<std::size_t>(a) * static_cast<std::size_t>(dim_b) *
               static_cast<std::size_t>(dim_c) +
           static_cast<std::size_t>(b) * static_cast<std::size_t>(dim_c) +
           static_cast<std::size_t>(c);
}

double noise_sxv_sum(
    const py::array_t<int, py::array::c_style | py::array::forcecast> &ix_arr,
    const py::array_t<int, py::array::c_style | py::array::forcecast> &jx_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &loadings_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &scores_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &sv_by_col_arr,
    const py::object &loading_covariances_obj,
    int num_cpu
) {
    const auto ix_buf = ix_arr.request();
    const auto jx_buf = jx_arr.request();
    const auto loadings_buf = loadings_arr.request();
    const auto scores_buf = scores_arr.request();
    const auto sv_buf = sv_by_col_arr.request();

    if (ix_buf.ndim != 1 || jx_buf.ndim != 1) {
        throw std::invalid_argument("ix and jx must be 1-D arrays.");
    }
    if (ix_buf.size != jx_buf.size) {
        throw std::invalid_argument("ix and jx must have the same length.");
    }

    if (loadings_buf.ndim != 2 || scores_buf.ndim != 2 || sv_buf.ndim != 3) {
        throw std::invalid_argument("loadings/scores/sv_by_col dimensions are invalid.");
    }

    const int n_features = static_cast<int>(loadings_buf.shape[0]);
    const int n_components = static_cast<int>(loadings_buf.shape[1]);
    const int n_samples = static_cast<int>(scores_buf.shape[1]);

    if (static_cast<int>(scores_buf.shape[0]) != n_components) {
        throw std::invalid_argument("scores.shape[0] must match loadings.shape[1].");
    }
    if (
        static_cast<int>(sv_buf.shape[0]) != n_samples ||
        static_cast<int>(sv_buf.shape[1]) != n_components ||
        static_cast<int>(sv_buf.shape[2]) != n_components
    ) {
        throw std::invalid_argument("sv_by_col must have shape (n_samples, k, k).");
    }

    const auto *ix_ptr = static_cast<const int *>(ix_buf.ptr);
    const auto *jx_ptr = static_cast<const int *>(jx_buf.ptr);
    const auto *a_ptr = static_cast<const double *>(loadings_buf.ptr);
    const auto *s_ptr = static_cast<const double *>(scores_buf.ptr);
    const auto *sv_ptr = static_cast<const double *>(sv_buf.ptr);

    const int n_obs = static_cast<int>(ix_buf.size);

    py::array_t<double, py::array::c_style | py::array::forcecast> av_arr;
    const double *av_ptr = nullptr;
    if (!loading_covariances_obj.is_none()) {
        av_arr = py::array_t<double, py::array::c_style | py::array::forcecast>(
            loading_covariances_obj
        );
        auto av_buf = av_arr.request();
        if (av_buf.ndim != 3) {
            throw std::invalid_argument("loading_covariances must be 3-D.");
        }
        if (
            static_cast<int>(av_buf.shape[0]) != n_features ||
            static_cast<int>(av_buf.shape[1]) != n_components ||
            static_cast<int>(av_buf.shape[2]) != n_components
        ) {
            throw std::invalid_argument("loading_covariances shape mismatch.");
        }
        av_ptr = static_cast<const double *>(av_buf.ptr);
    }

    for (int p = 0; p < n_obs; ++p) {
        const int i = ix_ptr[p];
        const int j = jx_ptr[p];
        if (i < 0 || i >= n_features || j < 0 || j >= n_samples) {
            throw std::invalid_argument("ix/jx entry out of range.");
        }
    }

    const int threads = resolve_num_threads(num_cpu, std::max(1, n_obs));
    std::vector<double> partials(static_cast<std::size_t>(threads), 0.0);

    std::exception_ptr worker_error;
    std::mutex error_mutex;

    auto worker = [&](int worker_id, int p_start, int p_end) {
        try {
            double local = 0.0;
            for (int p = p_start; p < p_end; ++p) {
                const int i = ix_ptr[p];
                const int j = jx_ptr[p];

                // a_i^T * Sv_j * a_i
                double sv_term = 0.0;
                for (int r = 0; r < n_components; ++r) {
                    const double a_r = a_ptr[static_cast<std::size_t>(i) *
                                             static_cast<std::size_t>(n_components) +
                                             static_cast<std::size_t>(r)];
                    if (a_r == 0.0) {
                        continue;
                    }
                    for (int c = 0; c < n_components; ++c) {
                        const double a_c = a_ptr[static_cast<std::size_t>(i) *
                                                 static_cast<std::size_t>(n_components) +
                                                 static_cast<std::size_t>(c)];
                        sv_term += a_r *
                                   sv_ptr[idx3(j, r, c, n_components, n_components)] *
                                   a_c;
                    }
                }
                local += sv_term;

                if (av_ptr != nullptr) {
                    // s_j^T * Av_i * s_j
                    double cov_term = 0.0;
                    for (int r = 0; r < n_components; ++r) {
                        const double s_r = s_ptr[static_cast<std::size_t>(r) *
                                                 static_cast<std::size_t>(n_samples) +
                                                 static_cast<std::size_t>(j)];
                        if (s_r == 0.0) {
                            continue;
                        }
                        for (int c = 0; c < n_components; ++c) {
                            const double s_c = s_ptr[static_cast<std::size_t>(c) *
                                                     static_cast<std::size_t>(n_samples) +
                                                     static_cast<std::size_t>(j)];
                            cov_term += s_r *
                                        av_ptr[idx3(i, r, c, n_components, n_components)] *
                                        s_c;
                        }
                    }

                    // trace(Sv_j * Av_i) == sum(Sv_j .* Av_i)
                    double trace_term = 0.0;
                    for (int r = 0; r < n_components; ++r) {
                        for (int c = 0; c < n_components; ++c) {
                            trace_term +=
                                sv_ptr[idx3(j, r, c, n_components, n_components)] *
                                av_ptr[idx3(i, r, c, n_components, n_components)];
                        }
                    }
                    local += cov_term + trace_term;
                }
            }
            partials[static_cast<std::size_t>(worker_id)] = local;
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
            worker(0, 0, n_obs);
        } else {
            std::vector<std::thread> pool;
            pool.reserve(static_cast<std::size_t>(threads));

            const int per_thread = n_obs / threads;
            const int remainder = n_obs % threads;

            int current = 0;
            for (int t = 0; t < threads; ++t) {
                const int extra = (t < remainder) ? 1 : 0;
                const int start = current;
                const int end = start + per_thread + extra;
                current = end;
                pool.emplace_back(worker, t, start, end);
            }

            for (auto &th : pool) {
                th.join();
            }
        }
    }

    if (worker_error) {
        std::rethrow_exception(worker_error);
    }

    double total = 0.0;
    for (double part : partials) {
        total += part;
    }
    return total;
}

}  // namespace

PYBIND11_MODULE(noise_update_kernels, m) {
    m.doc() = "Noise variance accumulation kernels for VB-PCA.";

    m.def(
        "noise_sxv_sum",
        &noise_sxv_sum,
        py::arg("ix"),
        py::arg("jx"),
        py::arg("loadings"),
        py::arg("scores"),
        py::arg("sv_by_col"),
        py::arg("loading_covariances") = py::none(),
        py::arg("num_cpu") = 0
    );
}
