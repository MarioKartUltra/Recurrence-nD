// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Abraham Joseph Okayli Masaryk
/*
 * File:   CrossGeneralizedRecurrencePlot.cpp
 */

#include "CrossGeneralizedRecurrencePlot.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

CrossGeneralizedRecurrencePlot::CrossGeneralizedRecurrencePlot(
    const std::vector<std::vector<double>>& X,
    const std::vector<std::vector<double>>& Y,
    double threshold_in, const std::string& norm_in)
    : threshold(threshold_in), norm(norm_in)
{
    if (X.empty() || X[0].empty() || Y.empty() || Y[0].empty())
        throw std::invalid_argument("CRN: empty input field");
    if (threshold < 0)
        throw std::invalid_argument("CRN: threshold must be non-negative");
    if (norm != "euclidean" && norm != "manhattan" && norm != "supremum")
        throw std::invalid_argument("CRN: norm must be euclidean/manhattan/supremum");
    for (const auto& row : X)
        if (row.size() != X[0].size())
            throw std::invalid_argument("CRN: non-rectangular X");
    for (const auto& row : Y)
        if (row.size() != Y[0].size())
            throw std::invalid_argument("CRN: non-rectangular Y");

    nx1 = static_cast<int>(X.size());
    ny1 = static_cast<int>(X[0].size());
    nx2 = static_cast<int>(Y.size());
    ny2 = static_cast<int>(Y[0].size());

    compute_tensor(X, Y);
}

void CrossGeneralizedRecurrencePlot::compute_tensor(
    const std::vector<std::vector<double>>& X,
    const std::vector<std::vector<double>>& Y)
{
    const std::size_t total = total_entries();
    try { tensor.assign(total, 0); }
    catch (const std::bad_alloc&) {
        std::cerr << "CRN: allocation of " << total << " bytes failed." << std::endl;
        throw;
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i1 = 0; i1 < nx1; ++i1) {
        for (int i2 = 0; i2 < ny1; ++i2) {
            double xi = X[i1][i2];
            for (int j1 = 0; j1 < nx2; ++j1) {
                for (int j2 = 0; j2 < ny2; ++j2) {
                    if (std::abs(xi - Y[j1][j2]) <= threshold) {
                        tensor[flat_index(i1, i2, j1, j2)] = 1;
                    }
                }
            }
        }
    }
}

double CrossGeneralizedRecurrencePlot::CRR() const {
    long long c = 0;
    const std::size_t n = tensor.size();
    #pragma omp parallel for reduction(+:c) schedule(static)
    for (std::size_t k = 0; k < n; ++k) c += tensor[k];
    return (n > 0) ? static_cast<double>(c) / static_cast<double>(n) : 0.0;
}

void CrossGeneralizedRecurrencePlot::compute_cross_degrees() {
    const int NX = get_NX();
    const int NY = get_NY();
    cross_deg_X.assign(NX, 0);
    cross_deg_Y.assign(NY, 0);
    nbr_Y_of_X.assign(NX, {});
    nbr_X_of_Y.assign(NY, {});

    // For each X-vertex u, list the Y-neighbours. Cross-degree of u is the
    // size of that list. The symmetric list for Y is built in a second pass
    // by transposition rather than a second tensor scan.
    #pragma omp parallel for schedule(dynamic)
    for (int u = 0; u < NX; ++u) {
        int i1 = u / ny1, i2 = u % ny1;
        auto& row = nbr_Y_of_X[u];
        row.reserve(32);
        for (int j1 = 0; j1 < nx2; ++j1) {
            for (int j2 = 0; j2 < ny2; ++j2) {
                if (tensor[flat_index(i1, i2, j1, j2)]) {
                    row.push_back(j1 * ny2 + j2);
                }
            }
        }
        cross_deg_X[u] = static_cast<long long>(row.size());
    }

    // Transpose to produce nbr_X_of_Y. Serial because of write pattern.
    for (int u = 0; u < NX; ++u) {
        for (int v : nbr_Y_of_X[u]) {
            nbr_X_of_Y[v].push_back(u);
        }
    }
    #pragma omp parallel for schedule(static)
    for (int v = 0; v < NY; ++v) {
        cross_deg_Y[v] = static_cast<long long>(nbr_X_of_Y[v].size());
    }

    degrees_built = true;
}

double CrossGeneralizedRecurrencePlot::jaccard(
    const std::vector<int>& a, const std::vector<int>& b) const
{
    int i = 0, j = 0, inter = 0;
    while (i < (int)a.size() && j < (int)b.size()) {
        if      (a[i] < b[j]) ++i;
        else if (a[i] > b[j]) ++j;
        else { ++inter; ++i; ++j; }
    }
    int uni = static_cast<int>(a.size() + b.size()) - inter;
    return (uni > 0) ? static_cast<double>(inter) / uni : 0.0;
}

double CrossGeneralizedRecurrencePlot::xmatch_XX(int u1, int u2) const {
    if (!degrees_built)
        throw std::runtime_error("CRN: call compute_cross_degrees() first");
    if (u1 == u2) return 1.0;
    return jaccard(nbr_Y_of_X[u1], nbr_Y_of_X[u2]);
}

double CrossGeneralizedRecurrencePlot::xmatch_YY(int v1, int v2) const {
    if (!degrees_built)
        throw std::runtime_error("CRN: call compute_cross_degrees() first");
    if (v1 == v2) return 1.0;
    return jaccard(nbr_X_of_Y[v1], nbr_X_of_Y[v2]);
}

void CrossGeneralizedRecurrencePlot::print_summary() const {
    std::cout << "CrossGeneralizedRecurrencePlot summary" << std::endl;
    std::cout << "  X shape       = " << nx1 << " x " << ny1 << std::endl;
    std::cout << "  Y shape       = " << nx2 << " x " << ny2 << std::endl;
    std::cout << "  tensor size   = " << total_entries() << " entries ("
              << (total_entries() / (1024.0 * 1024.0)) << " MB)" << std::endl;
    std::cout << "  threshold     = " << threshold << std::endl;
    std::cout << "  CRR           = " << CRR() << std::endl;
    if (degrees_built) {
        long long min_kX = cross_deg_X.empty() ? 0 : cross_deg_X[0];
        long long max_kX = min_kX;
        double mean_kX = 0.0;
        for (long long k : cross_deg_X) {
            if (k < min_kX) min_kX = k;
            if (k > max_kX) max_kX = k;
            mean_kX += k;
        }
        if (!cross_deg_X.empty())
            mean_kX /= static_cast<double>(cross_deg_X.size());
        std::cout << "  <k_X>         = " << mean_kX
                  << " (min " << min_kX << ", max " << max_kX << ")" << std::endl;
    }
}

void CrossGeneralizedRecurrencePlot::write_cross_degree_X_field(
    const std::string& filename) const
{
    std::ofstream out(filename);
    if (!out) throw std::runtime_error("CRN: cannot open " + filename);
    for (int i1 = 0; i1 < nx1; ++i1) {
        for (int i2 = 0; i2 < ny1; ++i2) {
            out << cross_deg_X[i1 * ny1 + i2];
            if (i2 + 1 < ny1) out << ' ';
        }
        out << '\n';
    }
}

void CrossGeneralizedRecurrencePlot::write_cross_degree_Y_field(
    const std::string& filename) const
{
    std::ofstream out(filename);
    if (!out) throw std::runtime_error("CRN: cannot open " + filename);
    for (int j1 = 0; j1 < nx2; ++j1) {
        for (int j2 = 0; j2 < ny2; ++j2) {
            out << cross_deg_Y[j1 * ny2 + j2];
            if (j2 + 1 < ny2) out << ' ';
        }
        out << '\n';
    }
}
