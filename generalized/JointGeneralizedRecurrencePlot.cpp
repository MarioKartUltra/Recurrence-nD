// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Abraham Joseph Okayli Masaryk
/*
 * File:   JointGeneralizedRecurrencePlot.cpp
 */

#include "JointGeneralizedRecurrencePlot.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

// Shape-validated tensor construction. Writes R^X * R^Y in one pass so we
// never materialise the intermediate per-field tensors separately.
std::vector<std::uint8_t> JointGeneralizedRecurrencePlot::build_joint_tensor(
    const std::vector<std::vector<double>>& X, double eps_X,
    const std::vector<std::vector<double>>& Y, double eps_Y,
    const std::string& norm)
{
    if (X.empty() || X[0].empty())
        throw std::invalid_argument("JRN: empty field X");
    if (Y.size() != X.size() || Y[0].size() != X[0].size())
        throw std::invalid_argument("JRN: X and Y must share nx and ny");
    if (eps_X < 0 || eps_Y < 0)
        throw std::invalid_argument("JRN: thresholds must be non-negative");
    if (norm != "euclidean" && norm != "manhattan" && norm != "supremum")
        throw std::invalid_argument("JRN: norm must be euclidean/manhattan/supremum");
    for (std::size_t i = 0; i < X.size(); ++i)
        if (X[i].size() != X[0].size() || Y[i].size() != X[0].size())
            throw std::invalid_argument("JRN: non-rectangular field");

    const int nx = static_cast<int>(X.size());
    const int ny = static_cast<int>(X[0].size());
    const std::size_t N4 = static_cast<std::size_t>(nx) * ny * nx * ny;

    std::vector<std::uint8_t> tensor;
    try { tensor.assign(N4, 0); }
    catch (const std::bad_alloc&) {
        std::cerr << "JRN: allocation of " << N4 << " bytes failed." << std::endl;
        throw;
    }

    auto flat = [nx, ny](int i1, int i2, int j1, int j2) {
        return ((static_cast<std::size_t>(i1) * ny + i2) * nx + j1) * ny + j2;
    };

    // For scalar fields the 'norm' argument collapses to |.| regardless of
    // choice, so we simply compare absolute differences. Using `!(dx <= eps)`
    // (instead of `dx > eps`) rejects NaN pairs correctly: NaN comparisons
    // return False under IEEE 754, so `!(NaN <= eps)` = True -> continue,
    // matching the scalar GRP's `d <= threshold` convention.
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i1 = 0; i1 < nx; ++i1) {
        for (int i2 = 0; i2 < ny; ++i2) {
            double xi = X[i1][i2];
            double yi = Y[i1][i2];
            for (int j1 = 0; j1 < nx; ++j1) {
                for (int j2 = 0; j2 < ny; ++j2) {
                    double dx = std::abs(xi - X[j1][j2]);
                    if (!(dx <= eps_X)) continue;
                    double dy = std::abs(yi - Y[j1][j2]);
                    if (!(dy <= eps_Y)) continue;
                    tensor[flat(i1, i2, j1, j2)] = 1;
                }
            }
        }
    }

    return tensor;
}

JointGeneralizedRecurrencePlot::JointGeneralizedRecurrencePlot(
    const std::vector<std::vector<double>>& X, double eps_X_in,
    const std::vector<std::vector<double>>& Y, double eps_Y_in,
    const std::string& norm)
    : eps_X(eps_X_in), eps_Y(eps_Y_in),
      // build_joint_tensor validates shapes and throws on mismatch, so the
      // dimensions passed to the GRP facade are guaranteed consistent.
      combined(build_joint_tensor(X, eps_X_in, Y, eps_Y_in, norm),
               static_cast<int>(X.size()),
               X.empty() ? 0 : static_cast<int>(X[0].size()),
               std::max(eps_X_in, eps_Y_in),
               norm)
{
}

void JointGeneralizedRecurrencePlot::print_summary() const {
    std::cout << "JointGeneralizedRecurrencePlot" << std::endl;
    std::cout << "  eps_X          = " << eps_X << std::endl;
    std::cout << "  eps_Y          = " << eps_Y << std::endl;
    combined.print_summary();
}
