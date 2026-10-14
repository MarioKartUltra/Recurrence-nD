// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Abraham Joseph Okayli Masaryk
/*
 * File:   SpatialRQA.cpp
 */

#include "SpatialRQA.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

SpatialRQA::SpatialRQA(const GeneralizedRecurrencePlot& grp_in,
                       int l_min_in, int v_min_in)
    : grp(grp_in), l_min(l_min_in), v_min(v_min_in),
      RR(0.0), DET_HS(0.0), LAM_HS(0.0), L_HS(0.0), TT_HS(0.0)
{
    if (l_min < 1) l_min = 1;
    if (v_min < 1) v_min = 1;
    calculate(l_min, v_min);
}

void SpatialRQA::calculate(int l_min_in, int v_min_in) {
    l_min = l_min_in;
    v_min = v_min_in;

    int M = max_block_size();
    Pl.assign(M + 1, 0);
    Pv.assign(M + 1, 0);

    compute_RR();
    compute_diagonal_histogram();
    compute_vertical_histogram();
    compute_scalar_measures();
}

int SpatialRQA::R_safe(int i1, int i2, int j1, int j2) const {
    return grp.R(i1, i2, j1, j2);
}

int SpatialRQA::R_bdry(int i1, int i2, int j1, int j2) const {
    // Convention: entries outside the tensor are treated as non-recurrent (0).
    // This permits blocks to "start" at the boundary and to "terminate" at
    // the boundary without requiring a strict non-recurrent neighbour.
    if (!in_range_i(i1) || !in_range_i(j1)) return 0;
    if (!in_range_j(i2) || !in_range_j(j2)) return 0;
    return grp.R(i1, i2, j1, j2);
}

int SpatialRQA::max_block_size() const {
    return std::min(grp.get_nx(), grp.get_ny());
}

void SpatialRQA::compute_RR() {
    std::size_t nrec = grp.total_recurrent_pairs();
    std::size_t ntot = grp.total_entries();
    RR = ntot > 0 ? static_cast<double>(nrec) / static_cast<double>(ntot) : 0.0;
}

/*
 * Diagonal hyper-surfaces:
 *   A diagonal HS of size l starts at (i1, i2, j1, j2) if
 *     (1) R_bdry(i1-1, i2-1, j1-1, j2-1) == 0   (diagonal left-up boundary)
 *     (2) for all k1,k2 in [0, l-1]: R(i1+k1, i2+k2, j1+k1, j2+k2) == 1
 *     (3) R_bdry(i1+l, i2+l, j1+l, j2+l) == 0   (diagonal right-down boundary)
 *
 * Algorithm: for each (i1, i2, j1, j2) candidate start, grow the largest
 * valid l. Increment Pl[l] if the termination condition (3) is met.
 */
void SpatialRQA::compute_diagonal_histogram() {
    const int nx = grp.get_nx();
    const int ny = grp.get_ny();
    const int Psize = static_cast<int>(Pl.size());

    // Each thread accumulates into a private histogram, then we reduce at the
    // end. Avoids atomic increments inside the hot inner loop.
    #pragma omp parallel
    {
        std::vector<long long> Pl_local(Psize, 0);

        #pragma omp for collapse(2) schedule(dynamic)
        for (int i1 = 0; i1 < nx; ++i1) {
            for (int i2 = 0; i2 < ny; ++i2) {
                for (int j1 = 0; j1 < nx; ++j1) {
                    for (int j2 = 0; j2 < ny; ++j2) {
                        // Exclude the hyper-surface of identity (analog of LOI in 1D RQA).
                        if (i1 == j1 && i2 == j2) continue;
                        if (R_safe(i1, i2, j1, j2) == 0) continue;
                        if (R_bdry(i1 - 1, i2 - 1, j1 - 1, j2 - 1) != 0) continue;

                        int l = 1;
                        int max_l = std::min({nx - i1, ny - i2,
                                              nx - j1, ny - j2});
                        while (l < max_l) {
                            bool ok = true;
                            for (int k2 = 0; k2 <= l && ok; ++k2) {
                                if (R_safe(i1 + l, i2 + k2, j1 + l, j2 + k2) == 0) ok = false;
                            }
                            if (ok) {
                                for (int k1 = 0; k1 < l && ok; ++k1) {
                                    if (R_safe(i1 + k1, i2 + l, j1 + k1, j2 + l) == 0) ok = false;
                                }
                            }
                            if (!ok) { break; }
                            ++l;
                        }
                        if (R_bdry(i1 + l, i2 + l, j1 + l, j2 + l) != 0) continue;
                        if (l >= 1 && l < Psize) {
                            Pl_local[l] += 1;
                        }
                    }
                }
            }
        }

        #pragma omp critical
        for (int l = 0; l < Psize; ++l) Pl[l] += Pl_local[l];
    }
}

/*
 * Vertical hyper-surfaces:
 *   A vertical HS of size v at (i1, i2, j1, j2) requires
 *     (1) R_bdry(i1, i2, j1-1, j2-1) == 0
 *     (2) for all k1, k2 in [0, v-1]: R(i1, i2, j1+k1, j2+k2) == 1
 *     (3) R_bdry(i1, i2, j1+v, j2+v) == 0
 *
 * Because (i1, i2) is fixed, we can treat this as finding maximal
 * diagonally-bounded squares in the 2D slice B[j1][j2] = R(i1, i2, j1, j2).
 * We run the growth algorithm for each anchor (i1, i2).
 */
void SpatialRQA::compute_vertical_histogram() {
    const int nx = grp.get_nx();
    const int ny = grp.get_ny();
    const int Psize = static_cast<int>(Pv.size());

    #pragma omp parallel
    {
        std::vector<long long> Pv_local(Psize, 0);

        #pragma omp for collapse(2) schedule(dynamic)
        for (int i1 = 0; i1 < nx; ++i1) {
            for (int i2 = 0; i2 < ny; ++i2) {
                // Slice at fixed (i1, i2): B[j1][j2] = R(i1, i2, j1, j2)
                for (int j1 = 0; j1 < nx; ++j1) {
                    for (int j2 = 0; j2 < ny; ++j2) {
                        if (R_safe(i1, i2, j1, j2) == 0) continue;
                        if (R_bdry(i1, i2, j1 - 1, j2 - 1) != 0) continue;

                        int v = 1;
                        int max_v = std::min(nx - j1, ny - j2);
                        while (v < max_v) {
                            bool ok = true;
                            for (int k2 = 0; k2 <= v && ok; ++k2) {
                                if (R_safe(i1, i2, j1 + v, j2 + k2) == 0) ok = false;
                            }
                            if (ok) {
                                for (int k1 = 0; k1 < v && ok; ++k1) {
                                    if (R_safe(i1, i2, j1 + k1, j2 + v) == 0) ok = false;
                                }
                            }
                            if (!ok) break;
                            ++v;
                        }
                        if (R_bdry(i1, i2, j1 + v, j2 + v) != 0) continue;
                        if (v >= 1 && v < Psize) {
                            Pv_local[v] += 1;
                        }
                    }
                }
            }
        }

        #pragma omp critical
        for (int v = 0; v < Psize; ++v) Pv[v] += Pv_local[v];
    }
}

/*
 * Scalar measures (Table 1 of Marwan 2007b):
 *   RR     = (1/N^(2d)) sum R
 *   DET_HS = sum_{l>=lmin} l * Pl[l] / sum R
 *   LAM_HS = sum_{v>=vmin} v * Pv[v] / sum_{v>=1} v * Pv[v]
 *   L_HS   = sum_{l>=lmin} l * Pl[l] / sum_{l>=lmin} Pl[l]
 *   TT_HS  = sum_{v>=vmin} v * Pv[v] / sum_{v>=vmin} Pv[v]
 *
 * Note: the normalisers differ between DET (total recurrent points) and
 * LAM (all vertical-block points of any size), following Marwan 2007b
 * Table 1 and the 1D convention.
 */
void SpatialRQA::compute_scalar_measures() {
    double sum_recurrent = static_cast<double>(grp.total_recurrent_pairs());

    // Diagonal-line sums
    double num_det = 0.0; // sum_{l>=lmin} l * Pl[l]
    double num_l   = 0.0; // sum_{l>=lmin} l * Pl[l]  (same as above)
    double den_l   = 0.0; // sum_{l>=lmin} Pl[l]
    for (int l = l_min; l < static_cast<int>(Pl.size()); ++l) {
        double lP = static_cast<double>(l) * Pl[l];
        num_det += lP;
        num_l   += lP;
        den_l   += Pl[l];
    }

    // Vertical-line sums
    double num_lam = 0.0; // sum_{v>=vmin} v * Pv[v]
    double den_lam = 0.0; // sum_{v>=1}    v * Pv[v]
    double num_tt  = 0.0; // sum_{v>=vmin} v * Pv[v]
    double den_tt  = 0.0; // sum_{v>=vmin} Pv[v]
    for (int v = 1; v < static_cast<int>(Pv.size()); ++v) {
        double vP = static_cast<double>(v) * Pv[v];
        den_lam += vP;
        if (v >= v_min) {
            num_lam += vP;
            num_tt  += vP;
            den_tt  += Pv[v];
        }
    }

    DET_HS = (sum_recurrent > 0) ? (num_det / sum_recurrent) : 0.0;
    LAM_HS = (den_lam > 0)       ? (num_lam / den_lam)       : 0.0;
    L_HS   = (den_l > 0)         ? (num_l / den_l)           : 0.0;
    TT_HS  = (den_tt > 0)        ? (num_tt / den_tt)         : 0.0;
}

void SpatialRQA::print_measures() const {
    std::cout << "SpatialRQA measures" << std::endl;
    std::cout << "  l_min = " << l_min << "    v_min = " << v_min << std::endl;
    std::cout << "  RR     = " << RR     << std::endl;
    std::cout << "  DET_HS = " << DET_HS << std::endl;
    std::cout << "  LAM_HS = " << LAM_HS << std::endl;
    std::cout << "  L_HS   = " << L_HS   << std::endl;
    std::cout << "  TT_HS  = " << TT_HS  << std::endl;
}

void SpatialRQA::write_measures(const std::string& filename) const {
    std::ofstream out(filename);
    if (!out) throw std::runtime_error("SpatialRQA: cannot open " + filename);
    out << "measure value\n";
    out << "RR "     << RR     << "\n";
    out << "DET_HS " << DET_HS << "\n";
    out << "LAM_HS " << LAM_HS << "\n";
    out << "L_HS "   << L_HS   << "\n";
    out << "TT_HS "  << TT_HS  << "\n";
    out << "l_min "  << l_min  << "\n";
    out << "v_min "  << v_min  << "\n";
}

void SpatialRQA::write_Pl(const std::string& filename) const {
    std::ofstream out(filename);
    if (!out) throw std::runtime_error("SpatialRQA: cannot open " + filename);
    out << "l Pl\n";
    for (std::size_t l = 1; l < Pl.size(); ++l)
        out << l << ' ' << Pl[l] << '\n';
}

void SpatialRQA::write_Pv(const std::string& filename) const {
    std::ofstream out(filename);
    if (!out) throw std::runtime_error("SpatialRQA: cannot open " + filename);
    out << "v Pv\n";
    for (std::size_t v = 1; v < Pv.size(); ++v)
        out << v << ' ' << Pv[v] << '\n';
}
