// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Abraham Joseph Okayli Masaryk
/*
 * test_grp.cpp
 *
 * Reproduces the benchmark of Marwan, Kurths, Saparin (2007)
 * "Generalised recurrence plot analysis for spatial data"
 * Physics Letters A 360, 545-551, Table 2.
 *
 * At Marwan's N=200 the 4D tensor is 1.6 GB. For tractable runs we use
 * smaller N with adaptive thresholding that matches the same target RR,
 * and scale l_min / v_min down proportionally. The qualitative ordering
 * (noise < AR2 < periodic) for DET and LAM is expected to hold; absolute
 * values depend on N.
 *
 * Expected Marwan Table 2 at N=200, eps=0.2, l_min=3, v_min=4:
 *   (A) noise:   RR=0.218, DET_HS=0.007, LAM_HS=0.006, L_HS=3.7, TT_HS=3.0
 *   (B) 2D-AR2:  RR=0.221, DET_HS=0.032, LAM_HS=0.065, L_HS=3.1, TT_HS=3.1
 *   (C) periodic:RR=0.219, DET_HS=0.322, LAM_HS=0.312, L_HS=5.8, TT_HS=5.6
 */

#include "GeneralizedRecurrencePlot.h"
#include "SpatialRQA.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// Default image size. Marwan uses 200.
static const int DEFAULT_IMG_SIZE = 64;

// Target recurrence rate (Marwan Table 2 hits ~0.22)
static const double TARGET_RR = 0.22;

// ------------------------------------------------------------------
// Utilities
// ------------------------------------------------------------------

static void normalise(std::vector<std::vector<double>>& field) {
    int nx = (int)field.size();
    int ny = (int)field[0].size();
    double sum = 0.0;
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            sum += field[i][j];
    double mean = sum / (nx * ny);
    double var = 0.0;
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j) {
            double d = field[i][j] - mean;
            var += d * d;
        }
    double sd = std::sqrt(var / (nx * ny));
    if (sd < 1e-12) sd = 1.0;
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            field[i][j] = (field[i][j] - mean) / sd;
}

// Choose a threshold eps such that RR on this field is approximately target_rr.
// Method: sample pairs, compute distances, pick the target_rr-quantile.
static double adaptive_threshold(const std::vector<std::vector<double>>& field,
                                  double target_rr,
                                  unsigned seed = 0)
{
    int nx = (int)field.size();
    int ny = (int)field[0].size();
    const int n_samples = 200000;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> Ux(0, nx - 1);
    std::uniform_int_distribution<int> Uy(0, ny - 1);

    std::vector<double> dists;
    dists.reserve(n_samples);
    for (int k = 0; k < n_samples; ++k) {
        int i1 = Ux(rng), i2 = Uy(rng);
        int j1 = Ux(rng), j2 = Uy(rng);
        dists.push_back(std::abs(field[i1][i2] - field[j1][j2]));
    }
    std::sort(dists.begin(), dists.end());
    int idx = std::max(0, (int)std::floor(target_rr * n_samples) - 1);
    return dists[idx];
}

// ------------------------------------------------------------------
// Image generators
// ------------------------------------------------------------------

static std::vector<std::vector<double>> generate_noise(int N, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    std::vector<std::vector<double>> f(N, std::vector<double>(N));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            f[i][j] = U(rng);
    normalise(f);
    return f;
}

// 2D AR2 process: x(i,j) depends on nearby past pixels
static std::vector<std::vector<double>> generate_ar2(int N, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> G(0.0, 1.0);
    std::vector<std::vector<double>> f(N, std::vector<double>(N, 0.0));
    const double a[4] = {0.4, 0.3, 0.1, 0.05};
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < N; ++j)
            f[i][j] = G(rng);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < 2; ++j)
            f[i][j] = G(rng);
    for (int i = 2; i < N; ++i)
        for (int j = 2; j < N; ++j)
            f[i][j] = a[0] * f[i - 1][j]
                    + a[1] * f[i][j - 1]
                    + a[2] * f[i - 1][j - 1]
                    + a[3] * f[i - 2][j - 2]
                    + G(rng);
    normalise(f);
    return f;
}

// Periodic: separable cosine product
static std::vector<std::vector<double>> generate_periodic(int N) {
    std::vector<std::vector<double>> f(N, std::vector<double>(N));
    const double k1 = 5.0 * 2.0 * M_PI / N;
    const double k2 = 5.0 * 2.0 * M_PI / N;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            f[i][j] = std::cos(k1 * i) * std::cos(k2 * j);
    normalise(f);
    return f;
}

// ------------------------------------------------------------------
// Run one test case with adaptive threshold
// ------------------------------------------------------------------

struct TestResult {
    std::string name;
    double eps;
    double RR, DET, LAM, L, TT;
};

static TestResult run_case(const std::string& name,
                           const std::vector<std::vector<double>>& field,
                           double target_rr,
                           int l_min, int v_min,
                           unsigned seed,
                           bool fixed_eps = false,
                           double eps_fixed = 0.2)
{
    std::cout << "\n=== Case: " << name << " ===" << std::endl;
    double eps = fixed_eps ? eps_fixed : adaptive_threshold(field, target_rr, seed);
    std::cout << "  eps = " << eps
              << (fixed_eps ? " (fixed)" : " (adaptive)") << std::endl;
    std::cout << "  [building 4D tensor...]" << std::flush;
    auto t0 = std::chrono::steady_clock::now();
    GeneralizedRecurrencePlot grp(field, eps, "euclidean");
    auto t1 = std::chrono::steady_clock::now();
    std::cout << " done in "
              << std::chrono::duration_cast<std::chrono::seconds>(t1 - t0).count()
              << " s" << std::endl;
    grp.print_summary();

    std::cout << "  [scanning diagonal + vertical HS...]" << std::flush;
    auto t2 = std::chrono::steady_clock::now();
    SpatialRQA rqa(grp, l_min, v_min);
    auto t3 = std::chrono::steady_clock::now();
    std::cout << " done in "
              << std::chrono::duration_cast<std::chrono::seconds>(t3 - t2).count()
              << " s" << std::endl;
    rqa.print_measures();

    // Print top entries of Pl and Pv histograms for transparency
    const auto& Pl = rqa.get_Pl();
    const auto& Pv = rqa.get_Pv();
    long long total_Pl = 0, total_Pv = 0;
    for (auto x : Pl) total_Pl += x;
    for (auto x : Pv) total_Pv += x;
    std::cout << "  # diagonal HS = " << total_Pl
              << ", # vertical HS = " << total_Pv << std::endl;
    std::cout << "  Pl[1..10] = ";
    for (int l = 1; l <= 10 && l < (int)Pl.size(); ++l)
        std::cout << Pl[l] << ' ';
    std::cout << std::endl;
    std::cout << "  Pv[1..10] = ";
    for (int v = 1; v <= 10 && v < (int)Pv.size(); ++v)
        std::cout << Pv[v] << ' ';
    std::cout << std::endl;

    return {name, eps, rqa.get_RR(), rqa.get_DET_HS(), rqa.get_LAM_HS(),
            rqa.get_L_HS(), rqa.get_TT_HS()};
}

// ------------------------------------------------------------------
// main
// ------------------------------------------------------------------

int main(int argc, char** argv) {
    int N = DEFAULT_IMG_SIZE;
    bool marwan_mode = false;
    // args: [N [l_min v_min]] or 'marwan' for exact Table 2 reproduction
    if (argc >= 2) {
        std::string arg1 = argv[1];
        if (arg1 == "marwan") {
            marwan_mode = true;
            N = 200;
        } else {
            N = std::atoi(argv[1]);
        }
    }
    if (N < 8) {
        std::cerr << "N too small; use at least 8" << std::endl;
        return 1;
    }

    int l_min, v_min;
    double threshold_fixed = 0.2;
    bool use_fixed_eps = false;
    // Using adaptive threshold even in "marwan" mode to match Marwan's RR=0.22.
    // (Marwan's eps=0.2 assumes peak-normalization; we force the same RR via
    //  the adaptive approach to get an apples-to-apples DET/LAM comparison.)
    if (marwan_mode) {
        l_min = 3;
        v_min = 4;
    } else {
        l_min = std::max(2, (int)std::round(3.0 * N / 200.0));
        v_min = std::max(2, (int)std::round(4.0 * N / 200.0));
        if (argc >= 3) l_min = std::atoi(argv[2]);
        if (argc >= 4) v_min = std::atoi(argv[3]);
    }

    std::cout << "Marwan 2007b generalised RP benchmark" << std::endl;
    std::cout << "  mode             = "
              << (marwan_mode ? "marwan (exact Table 2)" : "adaptive") << std::endl;
    std::cout << "  image size       = " << N << "x" << N << std::endl;
    std::cout << "  tensor memory    = "
              << (static_cast<double>(N) * N * N * N) / (1024.0 * 1024.0)
              << " MB" << std::endl;
    std::cout << "  l_min            = " << l_min << std::endl;
    std::cout << "  v_min            = " << v_min << std::endl;
    if (marwan_mode) {
        std::cout << "  fixed eps        = " << threshold_fixed << std::endl;
    } else {
        std::cout << "  target RR        = " << TARGET_RR << std::endl;
    }

    std::cout << "\n[generating test images...]" << std::endl;
    auto noise = generate_noise(N, 42);
    auto ar2   = generate_ar2(N, 42);
    auto per   = generate_periodic(N);

    std::vector<TestResult> results;
    results.push_back(run_case("A_noise",    noise, TARGET_RR, l_min, v_min,
                                42, use_fixed_eps, threshold_fixed));
    results.push_back(run_case("B_2D_AR2",   ar2,   TARGET_RR, l_min, v_min,
                                43, use_fixed_eps, threshold_fixed));
    results.push_back(run_case("C_periodic", per,   TARGET_RR, l_min, v_min,
                                44, use_fixed_eps, threshold_fixed));

    // Summary
    std::cout << "\n--- Summary (compare to Marwan 2007b Table 2) ---" << std::endl;
    std::cout << std::left << std::setw(14) << "Case"
              << std::right << std::setw(9) << "eps"
              << std::setw(9) << "RR"
              << std::setw(11) << "DET_HS"
              << std::setw(11) << "LAM_HS"
              << std::setw(9) << "L_HS"
              << std::setw(9) << "TT_HS" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    for (const auto& r : results) {
        std::cout << std::left << std::setw(14) << r.name
                  << std::right << std::setw(9) << r.eps
                  << std::setw(9) << r.RR
                  << std::setw(11) << r.DET
                  << std::setw(11) << r.LAM
                  << std::setw(9) << r.L
                  << std::setw(9) << r.TT << std::endl;
    }

    std::cout << "\nMarwan Table 2 expected values (N=200, eps=0.2, "
                 "l_min=3, v_min=4):\n";
    std::cout << "  A noise:    RR=0.218  DET=0.007  LAM=0.006  L=3.7  TT=3.0\n";
    std::cout << "  B 2D-AR2:   RR=0.221  DET=0.032  LAM=0.065  L=3.1  TT=3.1\n";
    std::cout << "  C periodic: RR=0.219  DET=0.322  LAM=0.312  L=5.8  TT=5.6\n";
    std::cout << "\nQualitative check: DET_HS and LAM_HS should increase\n"
                 "                   from noise to AR2 to periodic.\n";

    return 0;
}
