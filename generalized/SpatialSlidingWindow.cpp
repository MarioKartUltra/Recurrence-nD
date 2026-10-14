// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Abraham Joseph Okayli Masaryk
/*
 * File:   SpatialSlidingWindow.cpp
 */

#include "SpatialSlidingWindow.h"

#include "GeneralizedRecurrencePlot.h"
#include "SpatialRQA.h"
#include "SpatialRecurrenceNetwork.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

// Copy a w x h sub-field anchored at (ox, oy).
std::vector<std::vector<double>> extract_window(
    const std::vector<std::vector<double>>& field,
    int ox, int oy, int wx, int wy)
{
    std::vector<std::vector<double>> w(wx, std::vector<double>(wy));
    for (int i = 0; i < wx; ++i)
        for (int j = 0; j < wy; ++j)
            w[i][j] = field[ox + i][oy + j];
    return w;
}

double sample_quantile_abs_diff(
    const std::vector<std::vector<double>>& w,
    double q, std::mt19937& rng, int n_samples = 4000)
{
    int nx = (int)w.size();
    int ny = (int)w[0].size();
    std::uniform_int_distribution<int> Ux(0, nx - 1);
    std::uniform_int_distribution<int> Uy(0, ny - 1);
    std::vector<double> d;
    d.reserve(n_samples);
    for (int k = 0; k < n_samples; ++k) {
        int i1 = Ux(rng), i2 = Uy(rng);
        int j1 = Ux(rng), j2 = Uy(rng);
        d.push_back(std::abs(w[i1][i2] - w[j1][j2]));
    }
    std::sort(d.begin(), d.end());
    int idx = std::max(0, (int)std::floor(q * n_samples) - 1);
    return d[idx];
}

void allocate_fields(SpatialSlidingWindow::Result& r, bool net) {
    auto zero = [&](std::vector<std::vector<double>>& f) {
        f.assign(r.n_windows_x, std::vector<double>(r.n_windows_y, 0.0));
    };
    auto zeroI = [&](std::vector<std::vector<int>>& f) {
        f.assign(r.n_windows_x, std::vector<int>(r.n_windows_y, 0));
    };
    zeroI(r.origin_x);
    zeroI(r.origin_y);
    zero(r.RR); zero(r.DET_HS); zero(r.LAM_HS); zero(r.L_HS); zero(r.TT_HS);
    zero(r.eps_used);
    if (net) {
        zero(r.mean_degree); zero(r.mean_clustering);
        zero(r.transitivity); zero(r.assortativity);
    }
}

void validate(int nx, int ny, int wx, int wy, int sx, int sy) {
    if (wx < 4 || wy < 4)
        throw std::invalid_argument("SSW: window size must be >= 4");
    if (sx < 1 || sy < 1)
        throw std::invalid_argument("SSW: stride must be >= 1");
    if (wx > nx || wy > ny)
        throw std::invalid_argument("SSW: window larger than field");
}

}  // namespace

SpatialSlidingWindow::Result SpatialSlidingWindow::run_fixed_eps(
    const std::vector<std::vector<double>>& field,
    int wx, int wy, int stride_x, int stride_y,
    double eps, int l_min, int v_min,
    bool compute_network, const std::string& norm)
{
    if (field.empty() || field[0].empty())
        throw std::invalid_argument("SSW: empty field");
    int nx = (int)field.size();
    int ny = (int)field[0].size();
    validate(nx, ny, wx, wy, stride_x, stride_y);

    Result r;
    r.wx = wx; r.wy = wy;
    r.stride_x = stride_x; r.stride_y = stride_y;
    r.n_windows_x = (nx - wx) / stride_x + 1;
    r.n_windows_y = (ny - wy) / stride_y + 1;
    allocate_fields(r, compute_network);

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int wi = 0; wi < r.n_windows_x; ++wi) {
        for (int wj = 0; wj < r.n_windows_y; ++wj) {
            int ox = wi * stride_x;
            int oy = wj * stride_y;
            auto sub = extract_window(field, ox, oy, wx, wy);
            GeneralizedRecurrencePlot grp(sub, eps, norm);
            SpatialRQA rqa(grp, l_min, v_min);
            r.origin_x[wi][wj]   = ox;
            r.origin_y[wi][wj]   = oy;
            r.eps_used[wi][wj]   = eps;
            r.RR[wi][wj]         = rqa.get_RR();
            r.DET_HS[wi][wj]     = rqa.get_DET_HS();
            r.LAM_HS[wi][wj]     = rqa.get_LAM_HS();
            r.L_HS[wi][wj]       = rqa.get_L_HS();
            r.TT_HS[wi][wj]      = rqa.get_TT_HS();
            if (compute_network) {
                SpatialRecurrenceNetwork srn(grp);
                srn.compute_assortativity();
                r.mean_degree[wi][wj]     = srn.get_mean_degree();
                r.mean_clustering[wi][wj] = srn.get_mean_clustering();
                r.transitivity[wi][wj]    = srn.get_transitivity();
                r.assortativity[wi][wj]   = srn.get_assortativity();
            }
        }
    }
    return r;
}

SpatialSlidingWindow::Result SpatialSlidingWindow::run_target_rr(
    const std::vector<std::vector<double>>& field,
    int wx, int wy, int stride_x, int stride_y,
    double target_rr, int l_min, int v_min,
    bool compute_network, const std::string& norm, unsigned seed)
{
    if (field.empty() || field[0].empty())
        throw std::invalid_argument("SSW: empty field");
    int nx = (int)field.size();
    int ny = (int)field[0].size();
    validate(nx, ny, wx, wy, stride_x, stride_y);
    if (target_rr <= 0.0 || target_rr >= 1.0)
        throw std::invalid_argument("SSW: target_rr must be in (0, 1)");

    Result r;
    r.wx = wx; r.wy = wy;
    r.stride_x = stride_x; r.stride_y = stride_y;
    r.n_windows_x = (nx - wx) / stride_x + 1;
    r.n_windows_y = (ny - wy) / stride_y + 1;
    allocate_fields(r, compute_network);

    // Each thread gets its own RNG derived from the seed + thread id; avoids
    // sharing mt19937 state across threads.
    #pragma omp parallel
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        std::mt19937 rng(seed + 1000 * tid);

        #pragma omp for collapse(2) schedule(dynamic)
        for (int wi = 0; wi < r.n_windows_x; ++wi) {
            for (int wj = 0; wj < r.n_windows_y; ++wj) {
                int ox = wi * stride_x;
                int oy = wj * stride_y;
                auto sub = extract_window(field, ox, oy, wx, wy);
                double eps = sample_quantile_abs_diff(sub, target_rr, rng);
                GeneralizedRecurrencePlot grp(sub, eps, norm);
                SpatialRQA rqa(grp, l_min, v_min);
                r.origin_x[wi][wj]   = ox;
                r.origin_y[wi][wj]   = oy;
                r.eps_used[wi][wj]   = eps;
                r.RR[wi][wj]         = rqa.get_RR();
                r.DET_HS[wi][wj]     = rqa.get_DET_HS();
                r.LAM_HS[wi][wj]     = rqa.get_LAM_HS();
                r.L_HS[wi][wj]       = rqa.get_L_HS();
                r.TT_HS[wi][wj]      = rqa.get_TT_HS();
                if (compute_network) {
                    SpatialRecurrenceNetwork srn(grp);
                    srn.compute_assortativity();
                    r.mean_degree[wi][wj]     = srn.get_mean_degree();
                    r.mean_clustering[wi][wj] = srn.get_mean_clustering();
                    r.transitivity[wi][wj]    = srn.get_transitivity();
                    r.assortativity[wi][wj]   = srn.get_assortativity();
                }
            }
        }
    }
    return r;
}

namespace {
void write_grid(const std::vector<std::vector<double>>& g,
                const std::string& path)
{
    std::ofstream out(path);
    if (!out) throw std::runtime_error("SSW: cannot open " + path);
    for (const auto& row : g) {
        for (std::size_t j = 0; j < row.size(); ++j) {
            out << row[j];
            if (j + 1 < row.size()) out << ' ';
        }
        out << '\n';
    }
}
}  // namespace

void SpatialSlidingWindow::write_result(const Result& r,
                                        const std::string& prefix)
{
    write_grid(r.RR,             prefix + "_RR.dat");
    write_grid(r.DET_HS,         prefix + "_DET_HS.dat");
    write_grid(r.LAM_HS,         prefix + "_LAM_HS.dat");
    write_grid(r.L_HS,           prefix + "_L_HS.dat");
    write_grid(r.TT_HS,          prefix + "_TT_HS.dat");
    write_grid(r.eps_used,       prefix + "_eps.dat");
    if (!r.mean_degree.empty()) {
        write_grid(r.mean_degree,     prefix + "_mean_degree.dat");
        write_grid(r.mean_clustering, prefix + "_mean_clustering.dat");
        write_grid(r.transitivity,    prefix + "_transitivity.dat");
        write_grid(r.assortativity,   prefix + "_assortativity.dat");
    }
}
