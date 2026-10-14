// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Abraham Joseph Okayli Masaryk
/*
 * File:   GeneralizedRecurrencePlot.h
 * Purpose: 2D spatial Generalised Recurrence Plot per Marwan, Kurths, Saparin
 *          Physics Letters A 360 (2007) 545-551.
 *
 * For a 2D image X(i1, i2) with i1 in [0, nx), i2 in [0, ny), defines the 4D
 * recurrence tensor
 *     R(i1, i2, j1, j2) = Theta( eps - | X(i1,i2) - X(j1,j2) | )
 *
 * The tensor is stored flat as a vector<uint8_t> of length (nx*ny)^2.
 * Index layout (row-major):
 *     idx(i1, i2, j1, j2) = ((i1*ny + i2) * nx + j1) * ny + j2
 *
 * Memory: N^4 bytes for an NxN image. Practical limit ~256x256 (~4 GB).
 */

#ifndef GENERALIZEDRECURRENCEPLOT_H
#define GENERALIZEDRECURRENCEPLOT_H

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

class GeneralizedRecurrencePlot {
public:
    // Constructors
    GeneralizedRecurrencePlot(const std::vector<std::vector<double>>& field,
                              double threshold,
                              const std::string& norm = "euclidean");
    GeneralizedRecurrencePlot(const double* field_flat, int nx, int ny,
                              double threshold,
                              const std::string& norm = "euclidean");

    // Patch-embedding constructor (Marwan-style extension).
    //   Each pixel (i1, i2) becomes the vector
    //     x = (X(i1 + k1*tau1, i2 + k2*tau2))_{0 <= k1 < m1, 0 <= k2 < m2}
    //   and R = 1 iff ||x_i - x_j|| <= threshold in the chosen norm.
    //   Pixels within (m1-1)*tau1, (m2-1)*tau2 of the far boundary are
    //   excluded (their state would reach out of the field). The tensor
    //   shape matches the effective valid region (nx_eff, ny_eff).
    // Pass m1 = m2 = 1 (default of the value-only ctor) to disable embedding.
    GeneralizedRecurrencePlot(const std::vector<std::vector<double>>& field,
                              int m1, int m2, int tau1, int tau2,
                              double threshold,
                              const std::string& norm = "euclidean");

    // Adopt a pre-computed tensor. Used by Joint/Cross recurrence builders
    // and by sliding-window workflows that construct the tensor externally.
    // Tensor layout must match: ((i1*ny + i2)*nx + j1)*ny + j2.
    GeneralizedRecurrencePlot(std::vector<std::uint8_t> tensor,
                              int nx, int ny, double threshold,
                              const std::string& norm = "euclidean");

    // Vector-valued constructor: each pixel is a vector of length nchan.
    //   R(i1,i2,j1,j2) = Θ(ε − ||v(i1,i2) − v(j1,j2)||)  in the chosen norm.
    // Breaks the 1D-scalar-threshold clustering saturation (Donner 2010
    // Appendix A) by putting the state in a multi-dimensional space.
    // Input buffer layout is row-major (nx, ny, nchan).
    GeneralizedRecurrencePlot(const double* vector_field_flat,
                              int nx, int ny, int nchan,
                              double threshold,
                              const std::string& norm = "euclidean");
    GeneralizedRecurrencePlot(
        const std::vector<std::vector<std::vector<double>>>& vector_field,
        double threshold, const std::string& norm = "euclidean");

    // Getters
    int get_nx() const { return nx; }
    int get_ny() const { return ny; }
    double get_threshold() const { return threshold; }
    std::string get_norm() const { return norm; }
    std::size_t total_entries() const {
        return static_cast<std::size_t>(nx) * ny * nx * ny;
    }
    std::size_t total_recurrent_pairs() const;

    // Query single entry (0 or 1)
    inline int R(int i1, int i2, int j1, int j2) const {
        return tensor[flat_index(i1, i2, j1, j2)];
    }

    // Access raw storage (for SpatialRQA)
    const std::vector<std::uint8_t>& get_tensor() const { return tensor; }

    // I/O
    // 2D slice at fixed (i1, i2): a 2D image of size nx x ny
    void write_slice_fixed_i(int i1, int i2, const std::string& filename) const;
    // 3D subsection along HSOI: fix i2 = j2 = k, vary (i1, j1, k)
    void write_hsoi_subsection_fixed_i2_eq_j2(int k, const std::string& filename) const;
    // Sparse list of all recurrent quadruples
    void write_sparse_list(const std::string& filename) const;

    // Summary printing
    void print_summary() const;

private:
    int nx;
    int ny;
    double threshold;
    std::string norm;
    std::vector<std::uint8_t> tensor;
    std::vector<std::vector<double>> field;

    // Embedding parameters. (1, 1, 1, 1) = disabled, scalar-value mode.
    int m1{1}, m2{1}, tau1{1}, tau2{1};

    inline std::size_t flat_index(int i1, int i2, int j1, int j2) const {
        return ((static_cast<std::size_t>(i1) * ny + i2) * nx + j1) * ny + j2;
    }

    void compute_tensor();
    void compute_tensor_embedded();
    void compute_tensor_vector(const double* vfield, int nchan);
    double point_distance(double a, double b) const;
    double vector_distance(const std::vector<double>& a,
                           const std::vector<double>& b) const;
};

#endif /* GENERALIZEDRECURRENCEPLOT_H */
