// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Abraham Joseph Okayli Masaryk
/*
 * test_joint_cross_window.cpp
 *
 * Smoke test for the JRN/CRN 2D extensions, the sliding-window driver, and
 * the anisotropic patch-embedding constructor of GeneralizedRecurrencePlot.
 *
 * Cases
 * -----
 * 1. JRN: coupled fields (Y = X + small noise) should yield a joint RR close
 *    to the product of individual RRs if independent, but significantly higher
 *    if coupled. Here coupling pushes the joint RR above the independence
 *    baseline.
 *
 * 2. CRN: CRR of (X, copy(X)) should equal the auto-RR of X; CRR of
 *    (X, random_perm(X)) drops toward the independence baseline.
 *
 * 3. Sliding window: a field that is pure cosine on the left half and pure
 *    noise on the right half. DET_HS, mean_clustering, and transitivity
 *    should all be higher in the left windows than the right.
 *
 * 4. Anisotropic patch embedding: a cosine striped along the x-axis and
 *    flat along the y-axis. With embedding (m1=3, m2=1, tau1=1, tau2=1),
 *    we capture the structured direction; with (m1=1, m2=3) we embed only
 *    the flat direction and RR should saturate near 1. Asymmetry between
 *    the two cases is the anisotropy signature used in frozen-jet transect
 *    analyses.
 */

#include "GeneralizedRecurrencePlot.h"
#include "JointGeneralizedRecurrencePlot.h"
#include "CrossGeneralizedRecurrencePlot.h"
#include "SpatialSlidingWindow.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// ------------------------ generators ------------------------

static void normalise(std::vector<std::vector<double>>& f) {
    int nx = (int)f.size(), ny = (int)f[0].size();
    double s = 0.0;
    for (auto& r : f) for (double v : r) s += v;
    double mu = s / (nx * ny);
    double var = 0.0;
    for (auto& r : f) for (double v : r) { double d = v - mu; var += d * d; }
    double sd = std::sqrt(var / (nx * ny));
    if (sd < 1e-12) sd = 1.0;
    for (auto& r : f) for (double& v : r) v = (v - mu) / sd;
}

// Periodic with phase shift: same family as gen_periodic but with a random
// phase offset, so two instances with different seeds are uncorrelated.
static std::vector<std::vector<double>> gen_periodic_shifted(int N, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 2.0 * M_PI);
    double p1 = U(rng), p2 = U(rng);
    std::vector<std::vector<double>> f(N, std::vector<double>(N));
    const double k = 5.0 * 2.0 * M_PI / N;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            f[i][j] = std::cos(k * i + p1) * std::cos(k * j + p2);
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

// 1D-extruded (striped) field: cosine along x, constant along y
static std::vector<std::vector<double>> gen_stripes_x(int N) {
    std::vector<std::vector<double>> f(N, std::vector<double>(N));
    const double k = 5.0 * 2.0 * M_PI / N;
    for (int i = 0; i < N; ++i) {
        double v = std::cos(k * i);
        for (int j = 0; j < N; ++j) f[i][j] = v;
    }
    normalise(f);
    return f;
}

// Half-noise, half-periodic mix along the x-axis
static std::vector<std::vector<double>> gen_mixed(int N, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> G(0.0, 1.0);
    std::vector<std::vector<double>> f(N, std::vector<double>(N));
    const double k = 5.0 * 2.0 * M_PI / N;
    int half = N / 2;
    for (int i = 0; i < half; ++i)
        for (int j = 0; j < N; ++j)
            f[i][j] = std::cos(k * i) * std::cos(k * j);
    for (int i = half; i < N; ++i)
        for (int j = 0; j < N; ++j)
            f[i][j] = G(rng);
    normalise(f);
    return f;
}

// Coupled field: Y = X + small noise (controls coupling strength)
static std::vector<std::vector<double>> gen_coupled(
    const std::vector<std::vector<double>>& X, double sigma, unsigned seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> G(0.0, sigma);
    int N = (int)X.size();
    std::vector<std::vector<double>> Y(N, std::vector<double>(N));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) Y[i][j] = X[i][j] + G(rng);
    normalise(Y);
    return Y;
}

static double adaptive_eps(const std::vector<std::vector<double>>& f,
                            double target_rr, unsigned seed)
{
    int nx = (int)f.size(), ny = (int)f[0].size();
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> Ux(0, nx - 1), Uy(0, ny - 1);
    const int n_samples = 100000;
    std::vector<double> d; d.reserve(n_samples);
    for (int k = 0; k < n_samples; ++k) {
        int i1 = Ux(rng), i2 = Uy(rng), j1 = Ux(rng), j2 = Uy(rng);
        d.push_back(std::abs(f[i1][i2] - f[j1][j2]));
    }
    std::sort(d.begin(), d.end());
    int idx = std::max(0, (int)std::floor(target_rr * n_samples) - 1);
    return d[idx];
}

// ------------------------ individual tests ------------------------

static void test_jrn(int N) {
    std::cout << "\n=== JRN 2D: coupled vs independent ===\n";
    auto X = gen_periodic(N);
    auto Y_coupled     = gen_coupled(X, 0.1, 11);
    // gen_periodic_shifted with a different seed gives an uncorrelated field.
    auto Y_independent = gen_periodic_shifted(N, 913);
    normalise(Y_coupled);

    double eps_X = adaptive_eps(X, 0.05, 42);
    double eps_Y = adaptive_eps(Y_independent, 0.05, 43);
    double eps_C = adaptive_eps(Y_coupled, 0.05, 44);

    JointGeneralizedRecurrencePlot J_coupled(X, eps_X, Y_coupled,     eps_C);
    JointGeneralizedRecurrencePlot J_indep  (X, eps_X, Y_independent, eps_Y);

    double rr_coupled = (double)J_coupled.grp().total_recurrent_pairs()
                        / J_coupled.grp().total_entries();
    double rr_indep   = (double)J_indep.grp().total_recurrent_pairs()
                        / J_indep.grp().total_entries();

    std::cout << "  eps_X, eps_Y, eps_C  = " << eps_X << ", " << eps_Y
              << ", " << eps_C << "\n";
    std::cout << "  joint RR (coupled)   = " << rr_coupled << "\n";
    std::cout << "  joint RR (indep)     = " << rr_indep << "\n";
    std::cout << "  independence baseline ~ RR_X * RR_Y = 0.0025\n";
    std::cout << "  (expected: coupled >> indep ~ 0.0025)\n";
}

static void test_crn(int N) {
    std::cout << "\n=== CRN 2D: tensor construction + degree fields ===\n";
    // X against itself (same field) should recover the auto-RR. X against
    // a different-histogram field gives a different CRR. All scalar cross
    // statistics (CRR, cross-degree, xmatch via Y-sets) are histogram
    // invariants; structural discrimination between two Ys with the same
    // histogram requires line-structure measures on the cross tensor
    // (DET_CRP, LAM_CRP) which aren't implemented yet. Flagged below.
    auto X = gen_periodic(N);
    std::mt19937 rng(31337);
    std::uniform_real_distribution<double> U(-2.0, 2.0);
    std::vector<std::vector<double>> Yflat(N, std::vector<double>(N));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) Yflat[i][j] = U(rng);
    normalise(Yflat);

    double eps = adaptive_eps(X, 0.05, 42);

    CrossGeneralizedRecurrencePlot C_auto (X, X,     eps);
    CrossGeneralizedRecurrencePlot C_other(X, Yflat, eps);
    C_auto.compute_cross_degrees();
    C_other.compute_cross_degrees();

    std::cout << "  eps                 = " << eps << "\n";
    std::cout << "  CRR(X, X)           = " << C_auto.CRR()
              << "   (should match target_rr ~0.05)\n";
    std::cout << "  CRR(X, uniform noise)=" << C_other.CRR()
              << "   (different histogram -> different CRR)\n";

    auto degree_stats = [](const std::vector<long long>& k) {
        double mean = 0.0;
        for (auto v : k) mean += v;
        mean /= std::max(1, (int)k.size());
        double var = 0.0;
        long long kmax = 0;
        for (auto v : k) {
            double d = v - mean;
            var += d * d;
            if (v > kmax) kmax = v;
        }
        return std::tuple<double, double, long long>(mean, std::sqrt(var / std::max(1,(int)k.size())), kmax);
    };
    auto [m_a, sd_a, km_a] = degree_stats(C_auto.get_cross_degree_X());
    auto [m_b, sd_b, km_b] = degree_stats(C_other.get_cross_degree_X());
    std::cout << "  cross-deg X (auto):  mean=" << m_a << "  sd=" << sd_a
              << "  max=" << km_a << "\n";
    std::cout << "  cross-deg X (other): mean=" << m_b << "  sd=" << sd_b
              << "  max=" << km_b << "\n";
    std::cout << "  NOTE: distinguishing two Ys with the same histogram\n"
              << "        needs DET_CRP / LAM_CRP on the rectangular cross\n"
              << "        tensor. Not yet implemented; base tensor is in place.\n";
}

static void test_sliding_window(int N) {
    std::cout << "\n=== Sliding window: half-periodic / half-noise ===\n";
    int w = std::min(16, N / 2);
    int s = std::max(2, w / 4);
    auto field = gen_mixed(N, 123);
    auto t0 = std::chrono::steady_clock::now();
    auto r = SpatialSlidingWindow::run_target_rr(field, w, w, s, s,
                                                  0.05, 2, 2, true);
    auto t1 = std::chrono::steady_clock::now();
    std::cout << "  window = " << w << "x" << w
              << "  stride = " << s
              << "  grid = " << r.n_windows_x << "x" << r.n_windows_y
              << "  elapsed "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
              << " ms\n";

    // Split windows by whether their origin is in the periodic half or the
    // noise half.
    double sum_det_per = 0, sum_det_noi = 0;
    double sum_C_per   = 0, sum_C_noi   = 0;
    int n_per = 0, n_noi = 0;
    int half = N / 2;
    for (int i = 0; i < r.n_windows_x; ++i) {
        for (int j = 0; j < r.n_windows_y; ++j) {
            int ox = r.origin_x[i][j];
            bool periodic = (ox + r.wx <= half);
            bool noise    = (ox >= half);
            if (periodic) {
                sum_det_per += r.DET_HS[i][j];
                sum_C_per   += r.mean_clustering[i][j];
                ++n_per;
            } else if (noise) {
                sum_det_noi += r.DET_HS[i][j];
                sum_C_noi   += r.mean_clustering[i][j];
                ++n_noi;
            }
        }
    }
    std::cout << "  periodic windows: n=" << n_per
              << "  <DET_HS>=" << (n_per ? sum_det_per / n_per : 0)
              << "  <C>="      << (n_per ? sum_C_per   / n_per : 0) << "\n";
    std::cout << "  noise windows:    n=" << n_noi
              << "  <DET_HS>=" << (n_noi ? sum_det_noi / n_noi : 0)
              << "  <C>="      << (n_noi ? sum_C_noi   / n_noi : 0) << "\n";
    std::cout << "  (expected: periodic windows > noise windows in both)\n";

    SpatialSlidingWindow::write_result(r, "swnd_mixed");
}

static void test_anisotropic_embedding(int N) {
    std::cout << "\n=== Patch embedding: anisotropic diagnostic ===\n";
    auto stripes = gen_stripes_x(N);  // structured along x, flat along y
    double eps = 0.3;                 // fixed value to make RR comparable

    GeneralizedRecurrencePlot G_scalar(stripes, eps, "euclidean");
    GeneralizedRecurrencePlot G_embed_x(stripes, 3, 1, 1, 1, eps, "euclidean");
    GeneralizedRecurrencePlot G_embed_y(stripes, 1, 3, 1, 1, eps, "euclidean");

    auto rr = [](const GeneralizedRecurrencePlot& g) {
        return (double)g.total_recurrent_pairs()
               / g.total_entries();
    };

    std::cout << "  field: cosine stripes along x, constant along y\n";
    std::cout << "  eps                        = " << eps << "\n";
    std::cout << "  RR scalar                  = " << rr(G_scalar)  << "\n";
    std::cout << "  RR embed (m1=3, m2=1)      = " << rr(G_embed_x) << "\n";
    std::cout << "  RR embed (m1=1, m2=3)      = " << rr(G_embed_y) << "\n";
    std::cout << "  anisotropy ratio = RR(y-embed) / RR(x-embed) = "
              << rr(G_embed_y) / std::max(1e-12, rr(G_embed_x)) << "\n";
    std::cout << "  (expected: >1 here because the y-embedding tracks the\n"
                 "   flat-along-y direction where everything recurs, while\n"
                 "   the x-embedding resolves the structured direction)\n";
}

// ------------------------ main ------------------------

int main(int argc, char** argv) {
    int N = 32;
    if (argc >= 2) N = std::atoi(argv[1]);
    if (N < 8) { std::cerr << "N >= 8 please\n"; return 1; }

    std::cout << "JRN + CRN + sliding window + anisotropic embedding test\n";
    std::cout << "  N          = " << N << "\n";
#ifdef _OPENMP
    std::cout << "  OpenMP     = " << omp_get_max_threads() << " threads\n";
#else
    std::cout << "  OpenMP     = no\n";
#endif

    test_jrn(N);
    test_crn(N);
    test_sliding_window(N);
    test_anisotropic_embedding(N);

    std::cout << "\nAll four cases completed.\n";
    return 0;
}
