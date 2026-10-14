// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Abraham Joseph Okayli Masaryk
/*
 * File:   CrossGeneralizedRecurrencePlot.h
 * Purpose: 2D cross-recurrence plot between two scalar fields X (nx1 x ny1)
 *          and Y (nx2 x ny2), possibly of different sizes. Extends the
 *          1D CRP (Marwan/Romano/Thiel/Kurths 2007) to 2D spatial data.
 *
 * Definition:
 *   R^CRP(i1, i2, j1, j2) = Theta(eps - |X(i1,i2) - Y(j1,j2)|)
 *
 * Storage: rectangular tensor of shape (nx1, ny1, nx2, ny2), flat layout
 *   idx = ((i1*ny1 + i2) * nx2 + j1) * ny2 + j2
 *
 * This is NOT symmetric and does NOT fit the GeneralizedRecurrencePlot
 * facade (which assumes square tensors). Bipartite-network measures are
 * provided directly on this class.
 *
 * Measures:
 *   CRR                 = mean tensor value
 *   cross_degree_X(u)   = # Y-pixels that recur with X-pixel u
 *   cross_degree_Y(v)   = # X-pixels that recur with Y-pixel v
 *   cross_matching(u,v) = Jaccard of cross-neighbourhoods (for u in X, v in X
 *                         or both in Y, comparing their Y-neighbour sets or
 *                         X-neighbour sets respectively)
 */

#ifndef CROSSGENERALIZEDRECURRENCEPLOT_H
#define CROSSGENERALIZEDRECURRENCEPLOT_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class CrossGeneralizedRecurrencePlot {
public:
    CrossGeneralizedRecurrencePlot(
        const std::vector<std::vector<double>>& X,
        const std::vector<std::vector<double>>& Y,
        double threshold,
        const std::string& norm = "euclidean");

    // Getters
    int get_nx1() const { return nx1; }
    int get_ny1() const { return ny1; }
    int get_nx2() const { return nx2; }
    int get_ny2() const { return ny2; }
    int get_NX() const { return nx1 * ny1; }  // number of X-vertices
    int get_NY() const { return nx2 * ny2; }  // number of Y-vertices

    double get_threshold() const { return threshold; }
    std::size_t total_entries() const {
        return static_cast<std::size_t>(nx1) * ny1 * nx2 * ny2;
    }
    double CRR() const;

    inline int R(int i1, int i2, int j1, int j2) const {
        return tensor[flat_index(i1, i2, j1, j2)];
    }

    // Compute cross-degree fields (call before using getters).
    void compute_cross_degrees();
    const std::vector<long long>& get_cross_degree_X() const { return cross_deg_X; }
    const std::vector<long long>& get_cross_degree_Y() const { return cross_deg_Y; }

    // Cross-matching index.
    //   xmatch_XX(u1, u2): Jaccard of the Y-neighbour sets of two X-pixels
    //   xmatch_YY(v1, v2): Jaccard of the X-neighbour sets of two Y-pixels
    double xmatch_XX(int u1, int u2) const;
    double xmatch_YY(int v1, int v2) const;

    // I/O
    void print_summary() const;
    void write_cross_degree_X_field(const std::string& filename) const;
    void write_cross_degree_Y_field(const std::string& filename) const;

private:
    int nx1, ny1;
    int nx2, ny2;
    double threshold;
    std::string norm;
    std::vector<std::uint8_t> tensor;

    // Per-vertex cross-neighbour sets (built by compute_cross_degrees).
    std::vector<std::vector<int>> nbr_Y_of_X;   // for each X-vertex
    std::vector<std::vector<int>> nbr_X_of_Y;   // for each Y-vertex
    std::vector<long long> cross_deg_X;
    std::vector<long long> cross_deg_Y;
    bool degrees_built{false};

    inline std::size_t flat_index(int i1, int i2, int j1, int j2) const {
        return ((static_cast<std::size_t>(i1) * ny1 + i2)
                * nx2 + j1) * ny2 + j2;
    }

    void compute_tensor(const std::vector<std::vector<double>>& X,
                        const std::vector<std::vector<double>>& Y);

    double jaccard(const std::vector<int>& a,
                   const std::vector<int>& b) const;
};

#endif /* CROSSGENERALIZEDRECURRENCEPLOT_H */
