// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Abraham Joseph Okayli Masaryk
/*
 * File:   SpatialRQA.h
 * Purpose: Generalised RQA measures for a 2D spatial recurrence plot,
 *          following Marwan, Kurths, Saparin (2007) Physics Letters A 360.
 *
 * Implements:
 *   RR     = recurrence rate                       (Table 1)
 *   DET_HS = determinism using diagonal hyper-surfaces (Eq. 8)
 *   LAM_HS = laminarity using vertical hyper-surfaces  (Eq. 10)
 *   L_HS   = averaged diagonal hyper-surface size
 *   TT_HS  = trapping size
 *
 * Diagonal HS of size l at (i1,i2,j1,j2):
 *   (1 - R(i1-1, i2-1, j1-1, j2-1)) (1 - R(i1+l, i2+l, j1+l, j2+l))
 *    prod_{k1,k2=0..l-1} R(i1+k1, i2+k2, j1+k1, j2+k2) = 1
 *
 * Vertical HS of size v at (i1,i2,j1,j2):
 *   (1 - R(i1, i2, j1-1, j2-1)) (1 - R(i1, i2, j1+v, j2+v))
 *    prod_{k1,k2=0..v-1} R(i1, i2, j1+k1, j2+k2) = 1
 *
 * The first product counts blocks bounded diagonally by non-recurrent points;
 * the second counts blocks anchored at a fixed (i1, i2) reference.
 */

#ifndef SPATIALRQA_H
#define SPATIALRQA_H

#include "GeneralizedRecurrencePlot.h"

#include <string>
#include <vector>

class SpatialRQA {
public:
    SpatialRQA(const GeneralizedRecurrencePlot& grp, int l_min, int v_min);

    // Calculation (called in constructor; recall with different thresholds if desired)
    void calculate(int l_min, int v_min);

    // Getters for the scalar measures
    double get_RR()     const { return RR; }
    double get_DET_HS() const { return DET_HS; }
    double get_LAM_HS() const { return LAM_HS; }
    double get_L_HS()   const { return L_HS; }
    double get_TT_HS()  const { return TT_HS; }

    // Getters for the size distributions
    const std::vector<long long>& get_Pl() const { return Pl; }
    const std::vector<long long>& get_Pv() const { return Pv; }

    // Getters for scan parameters
    int get_l_min() const { return l_min; }
    int get_v_min() const { return v_min; }

    // Printing / saving
    void print_measures() const;
    void write_measures(const std::string& filename) const;
    void write_Pl(const std::string& filename) const;
    void write_Pv(const std::string& filename) const;

private:
    const GeneralizedRecurrencePlot& grp;
    int l_min;
    int v_min;

    // Size-distribution histograms
    std::vector<long long> Pl; // Pl[l] = number of diagonal HS of size l
    std::vector<long long> Pv; // Pv[v] = number of vertical HS of size v

    // Scalar measures
    double RR;
    double DET_HS;
    double LAM_HS;
    double L_HS;
    double TT_HS;

    // Helpers
    bool in_range_i(int i) const { return i >= 0 && i < grp.get_nx(); }
    bool in_range_j(int j) const { return j >= 0 && j < grp.get_ny(); }
    int  R_safe(int i1, int i2, int j1, int j2) const;
    int  R_bdry(int i1, int i2, int j1, int j2) const; // outside array treated as 0

    void compute_RR();
    void compute_diagonal_histogram();
    void compute_vertical_histogram();
    void compute_scalar_measures();

    // Max possible block size in a 2D RP slice.
    // Used for memory allocation of histograms.
    int max_block_size() const;
};

#endif /* SPATIALRQA_H */
