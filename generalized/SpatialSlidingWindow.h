// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Abraham Joseph Okayli Masaryk
/*
 * File:   SpatialSlidingWindow.h
 * Purpose: Sliding-window 2D RQA and recurrence-network analysis. Scans a
 *          field with an (wx, wy) window stepped by (stride_x, stride_y)
 *          and computes SpatialRQA + SpatialRecurrenceNetwork summaries
 *          per window. Output is a stack of 2D "measure fields" of shape
 *          (n_windows_x, n_windows_y).
 *
 * Complexity per window: O(w^4) for the tensor plus scans.
 * Total: O((nx/stride) * (ny/stride) * w^4). Linear in the number of
 * windows, so OpenMP parallelises the outer window loop.
 *
 * Threshold selection:
 *   - run_fixed_eps(): one threshold used for every window.
 *   - run_target_rr(): per-window adaptive threshold chosen via sampling
 *     so each window has approximately the target recurrence rate.
 *
 * Measures stored per window (each a 2D grid of size n_windows_x x n_windows_y):
 *   RR, DET_HS, LAM_HS, L_HS, TT_HS,
 *   mean_degree, mean_clustering, transitivity, assortativity
 */

#ifndef SPATIALSLIDINGWINDOW_H
#define SPATIALSLIDINGWINDOW_H

#include <string>
#include <vector>

class SpatialSlidingWindow {
public:
    struct Result {
        int wx{0}, wy{0};
        int stride_x{1}, stride_y{1};
        int n_windows_x{0}, n_windows_y{0};

        // Origin (top-left pixel) of each window in the original field.
        std::vector<std::vector<int>> origin_x;  // [nwx][nwy]
        std::vector<std::vector<int>> origin_y;

        // SpatialRQA measures
        std::vector<std::vector<double>> RR;
        std::vector<std::vector<double>> DET_HS;
        std::vector<std::vector<double>> LAM_HS;
        std::vector<std::vector<double>> L_HS;
        std::vector<std::vector<double>> TT_HS;

        // Recurrence-network measures (populated iff compute_network=true)
        std::vector<std::vector<double>> mean_degree;
        std::vector<std::vector<double>> mean_clustering;
        std::vector<std::vector<double>> transitivity;
        std::vector<std::vector<double>> assortativity;

        // Per-window threshold actually used (useful for adaptive-RR mode)
        std::vector<std::vector<double>> eps_used;
    };

    // Fixed threshold across all windows.
    static Result run_fixed_eps(const std::vector<std::vector<double>>& field,
                                int wx, int wy,
                                int stride_x, int stride_y,
                                double eps,
                                int l_min, int v_min,
                                bool compute_network = true,
                                const std::string& norm = "euclidean");

    // Per-window adaptive threshold targeting a given recurrence rate.
    // Useful when the field's local variance changes across the scan.
    static Result run_target_rr(const std::vector<std::vector<double>>& field,
                                int wx, int wy,
                                int stride_x, int stride_y,
                                double target_rr,
                                int l_min, int v_min,
                                bool compute_network = true,
                                const std::string& norm = "euclidean",
                                unsigned seed = 42);

    // Write every measure field to text files named <prefix>_<measure>.dat.
    static void write_result(const Result& r, const std::string& prefix);
};

#endif /* SPATIALSLIDINGWINDOW_H */
