// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Abraham Joseph Okayli Masaryk
/*
 * test_network.cpp
 *
 * Smoke test for SpatialRecurrenceNetwork on the same three benchmark
 * fields used by test_grp.cpp (Marwan 2007b Table 2: noise, 2D-AR2,
 * periodic). Prints basic network summaries and writes spatial fields
 * of degree / clustering to disk for comparison with the underlying
 * recurrence structure.
 *
 * Expected qualitative behaviour:
 *   - noise:    low clustering, narrow degree distribution
 *   - 2D-AR2:   mild increase in clustering and degree spread
 *   - periodic: high clustering, strongly bimodal degree field
 */

#include "GeneralizedRecurrencePlot.h"
#include "SpatialRecurrenceNetwork.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

static const int    DEFAULT_N  = 32;
// Donner 2010 uses RR ~0.01 for network analysis; at higher RR the scalar-
// threshold structure forces very high clustering for every signal (see
// appendix A of Donner 2010). Keep it low so structural differences show up.
static const double TARGET_RR  = 0.03;

// ---- image generators (duplicated small helpers; see test_grp.cpp) -------

static void normalise(std::vector<std::vector<double>>& f) {
    int nx = (int)f.size(), ny = (int)f[0].size();
    double s = 0.0;
    for (auto& row : f) for (double v : row) s += v;
    double mu = s / (nx * ny);
    double var = 0.0;
    for (auto& row : f) for (double v : row) { double d = v - mu; var += d * d; }
    double sd = std::sqrt(var / (nx * ny));
    if (sd < 1e-12) sd = 1.0;
    for (auto& row : f) for (double& v : row) v = (v - mu) / sd;
}

static double adaptive_threshold(const std::vector<std::vector<double>>& f,
                                  double target_rr, unsigned seed)
{
    int nx = (int)f.size(), ny = (int)f[0].size();
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> Ux(0, nx - 1), Uy(0, ny - 1);
    const int n_samples = 200000;
    std::vector<double> d; d.reserve(n_samples);
    for (int k = 0; k < n_samples; ++k) {
        int i1 = Ux(rng), i2 = Uy(rng), j1 = Ux(rng), j2 = Uy(rng);
        d.push_back(std::abs(f[i1][i2] - f[j1][j2]));
    }
    std::sort(d.begin(), d.end());
    int idx = std::max(0, (int)std::floor(target_rr * n_samples) - 1);
    return d[idx];
}

static std::vector<std::vector<double>> gen_noise(int N, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    std::vector<std::vector<double>> f(N, std::vector<double>(N));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) f[i][j] = U(rng);
    normalise(f);
    return f;
}

static std::vector<std::vector<double>> gen_ar2(int N, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> G(0.0, 1.0);
    std::vector<std::vector<double>> f(N, std::vector<double>(N, 0.0));
    const double a[4] = {0.4, 0.3, 0.1, 0.05};
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < N; ++j) f[i][j] = G(rng);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < 2; ++j) f[i][j] = G(rng);
    for (int i = 2; i < N; ++i)
        for (int j = 2; j < N; ++j)
            f[i][j] = a[0] * f[i-1][j] + a[1] * f[i][j-1]
                    + a[2] * f[i-1][j-1] + a[3] * f[i-2][j-2] + G(rng);
    normalise(f);
    return f;
}

static std::vector<std::vector<double>> gen_periodic(int N) {
    std::vector<std::vector<double>> f(N, std::vector<double>(N));
    const double k = 5.0 * 2.0 * M_PI / N;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) f[i][j] = std::cos(k * i) * std::cos(k * j);
    normalise(f);
    return f;
}

// ---- run one case ---------------------------------------------------------

struct NetResult {
    std::string name;
    double eps;
    double edge_density;
    double mean_k;
    double mean_C;
    double transitivity;
    double assortativity;
};

static NetResult run_case(const std::string& name,
                          const std::vector<std::vector<double>>& field,
                          double target_rr,
                          unsigned seed,
                          bool do_centrality)
{
    std::cout << "\n=== " << name << " ===" << std::endl;
    double eps = adaptive_threshold(field, target_rr, seed);
    std::cout << "  eps = " << eps << std::endl;

    std::cout << "  [build 4D tensor]... " << std::flush;
    auto t0 = std::chrono::steady_clock::now();
    GeneralizedRecurrencePlot grp(field, eps, "euclidean");
    auto t1 = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
              << " ms" << std::endl;

    std::cout << "  [basic network measures]... " << std::flush;
    auto t2 = std::chrono::steady_clock::now();
    SpatialRecurrenceNetwork srn(grp);
    auto t3 = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()
              << " ms" << std::endl;

    std::cout << "  [assortativity]... " << std::flush;
    auto t4 = std::chrono::steady_clock::now();
    srn.compute_assortativity();
    auto t5 = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count()
              << " ms" << std::endl;

    if (do_centrality) {
        std::cout << "  [closeness]... " << std::flush;
        auto t6 = std::chrono::steady_clock::now();
        srn.compute_closeness();
        auto t7 = std::chrono::steady_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t7 - t6).count()
                  << " ms" << std::endl;

        std::cout << "  [betweenness]... " << std::flush;
        auto t8 = std::chrono::steady_clock::now();
        srn.compute_betweenness();
        auto t9 = std::chrono::steady_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t9 - t8).count()
                  << " ms" << std::endl;
    }

    srn.print_summary();

    // Drop the spatial fields next to the executable.
    srn.write_degree_field("field_degree_" + name + ".dat");
    srn.write_clustering_field("field_clustering_" + name + ".dat");
    if (do_centrality) {
        srn.write_closeness_field("field_closeness_" + name + ".dat");
        srn.write_betweenness_field("field_betweenness_" + name + ".dat");
    }

    return {name, eps,
            srn.get_edge_density(),
            srn.get_mean_degree(),
            srn.get_mean_clustering(),
            srn.get_transitivity(),
            srn.get_assortativity()};
}

// --------------------------------------------------------------------------

int main(int argc, char** argv) {
    int N = DEFAULT_N;
    bool do_centrality = false;
    if (argc >= 2) N = std::atoi(argv[1]);
    if (argc >= 3 && std::string(argv[2]) == "full") do_centrality = true;
    if (N < 8) { std::cerr << "N must be >= 8\n"; return 1; }

    std::cout << "SpatialRecurrenceNetwork smoke test" << std::endl;
    std::cout << "  N (image side)   = " << N << std::endl;
    std::cout << "  vertices         = " << N * N << std::endl;
    std::cout << "  target RR        = " << TARGET_RR << std::endl;
    std::cout << "  full centrality  = " << (do_centrality ? "yes" : "no")
              << std::endl;
#ifdef _OPENMP
    std::cout << "  OpenMP threads   = " << omp_get_max_threads() << std::endl;
#else
    std::cout << "  OpenMP threads   = 1 (compiled without OpenMP)" << std::endl;
#endif

    auto noise = gen_noise(N, 42);
    auto ar2   = gen_ar2(N, 42);
    auto per   = gen_periodic(N);

    std::vector<NetResult> results;
    results.push_back(run_case("noise",    noise, TARGET_RR, 42, do_centrality));
    results.push_back(run_case("ar2",      ar2,   TARGET_RR, 43, do_centrality));
    results.push_back(run_case("periodic", per,   TARGET_RR, 44, do_centrality));

    std::cout << "\n--- Summary ---\n";
    std::cout << std::left << std::setw(10) << "case"
              << std::right << std::setw(9)  << "eps"
              << std::setw(10) << "rho"
              << std::setw(10) << "<k>"
              << std::setw(10) << "<C>"
              << std::setw(10) << "T"
              << std::setw(10) << "r" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    for (const auto& r : results) {
        std::cout << std::left << std::setw(10) << r.name
                  << std::right << std::setw(9)  << r.eps
                  << std::setw(10) << r.edge_density
                  << std::setw(10) << r.mean_k
                  << std::setw(10) << r.mean_C
                  << std::setw(10) << r.transitivity
                  << std::setw(10) << r.assortativity << std::endl;
    }
    std::cout << "\nExpected trend (Donner 2010): <C> and T should increase\n"
                 "                               from noise to ar2 to periodic.\n";

    return 0;
}
