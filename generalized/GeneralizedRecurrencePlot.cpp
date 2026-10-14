// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Abraham Joseph Okayli Masaryk
/*
 * File:   GeneralizedRecurrencePlot.cpp
 */

#include "GeneralizedRecurrencePlot.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

GeneralizedRecurrencePlot::GeneralizedRecurrencePlot(
    const std::vector<std::vector<double>>& field_in,
    double threshold_in,
    const std::string& norm_in)
    : threshold(threshold_in), norm(norm_in), field(field_in)
{
    if (field.empty() || field[0].empty()) {
        throw std::invalid_argument("GRP: empty input field");
    }
    nx = static_cast<int>(field.size());
    ny = static_cast<int>(field[0].size());
    for (const auto& row : field) {
        if (static_cast<int>(row.size()) != ny) {
            throw std::invalid_argument("GRP: non-rectangular field");
        }
    }
    if (threshold < 0) {
        throw std::invalid_argument("GRP: threshold must be non-negative");
    }
    if (norm != "euclidean" && norm != "manhattan" && norm != "supremum") {
        throw std::invalid_argument("GRP: norm must be euclidean, manhattan or supremum");
    }
    compute_tensor();
}

GeneralizedRecurrencePlot::GeneralizedRecurrencePlot(
    const double* field_flat, int nx_in, int ny_in,
    double threshold_in, const std::string& norm_in)
    : nx(nx_in), ny(ny_in), threshold(threshold_in), norm(norm_in)
{
    if (nx <= 0 || ny <= 0) {
        throw std::invalid_argument("GRP: nx and ny must be positive");
    }
    field.resize(nx, std::vector<double>(ny));
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            field[i][j] = field_flat[i * ny + j];
    compute_tensor();
}

GeneralizedRecurrencePlot::GeneralizedRecurrencePlot(
    const std::vector<std::vector<double>>& field_in,
    int m1_in, int m2_in, int tau1_in, int tau2_in,
    double threshold_in, const std::string& norm_in)
    : threshold(threshold_in), norm(norm_in), field(field_in),
      m1(m1_in), m2(m2_in), tau1(tau1_in), tau2(tau2_in)
{
    if (field.empty() || field[0].empty())
        throw std::invalid_argument("GRP: empty input field");
    if (m1 < 1 || m2 < 1 || tau1 < 1 || tau2 < 1)
        throw std::invalid_argument("GRP: embedding m1,m2,tau1,tau2 must be >= 1");
    if (threshold < 0)
        throw std::invalid_argument("GRP: threshold must be non-negative");
    if (norm != "euclidean" && norm != "manhattan" && norm != "supremum")
        throw std::invalid_argument("GRP: norm must be euclidean, manhattan or supremum");

    int input_nx = static_cast<int>(field.size());
    int input_ny = static_cast<int>(field[0].size());
    for (const auto& row : field)
        if (static_cast<int>(row.size()) != input_ny)
            throw std::invalid_argument("GRP: non-rectangular field");

    // The effective grid is the set of (i1, i2) for which the full patch fits
    // inside the field. Shrink nx, ny accordingly; the tensor is built over
    // the valid region only.
    nx = input_nx - (m1 - 1) * tau1;
    ny = input_ny - (m2 - 1) * tau2;
    if (nx <= 0 || ny <= 0)
        throw std::invalid_argument("GRP: embedding larger than field");

    compute_tensor_embedded();
}

GeneralizedRecurrencePlot::GeneralizedRecurrencePlot(
    std::vector<std::uint8_t> tensor_in,
    int nx_in, int ny_in, double threshold_in, const std::string& norm_in)
    : nx(nx_in), ny(ny_in), threshold(threshold_in), norm(norm_in),
      tensor(std::move(tensor_in))
{
    if (nx <= 0 || ny <= 0)
        throw std::invalid_argument("GRP: nx and ny must be positive");
    std::size_t expected = static_cast<std::size_t>(nx) * ny * nx * ny;
    if (tensor.size() != expected)
        throw std::invalid_argument(
            "GRP: adopted tensor size does not match nx*ny*nx*ny");
}

GeneralizedRecurrencePlot::GeneralizedRecurrencePlot(
    const double* vfield_flat, int nx_in, int ny_in, int nchan,
    double threshold_in, const std::string& norm_in)
    : nx(nx_in), ny(ny_in), threshold(threshold_in), norm(norm_in)
{
    if (nx <= 0 || ny <= 0)
        throw std::invalid_argument("GRP: nx and ny must be positive");
    if (nchan < 1)
        throw std::invalid_argument("GRP: nchan must be >= 1");
    if (threshold < 0)
        throw std::invalid_argument("GRP: threshold must be non-negative");
    if (norm != "euclidean" && norm != "manhattan" && norm != "supremum")
        throw std::invalid_argument("GRP: norm must be euclidean/manhattan/supremum");
    // The scalar `field` member stays empty; the vector computation reads the
    // external buffer directly, so we skip the copy.
    compute_tensor_vector(vfield_flat, nchan);
}

GeneralizedRecurrencePlot::GeneralizedRecurrencePlot(
    const std::vector<std::vector<std::vector<double>>>& vector_field,
    double threshold_in, const std::string& norm_in)
    : threshold(threshold_in), norm(norm_in)
{
    if (vector_field.empty() || vector_field[0].empty() || vector_field[0][0].empty())
        throw std::invalid_argument("GRP: empty vector field");
    if (threshold < 0)
        throw std::invalid_argument("GRP: threshold must be non-negative");
    if (norm != "euclidean" && norm != "manhattan" && norm != "supremum")
        throw std::invalid_argument("GRP: norm must be euclidean/manhattan/supremum");

    nx = static_cast<int>(vector_field.size());
    ny = static_cast<int>(vector_field[0].size());
    int nchan = static_cast<int>(vector_field[0][0].size());
    for (const auto& row : vector_field) {
        if (static_cast<int>(row.size()) != ny)
            throw std::invalid_argument("GRP: non-rectangular vector field");
        for (const auto& v : row) {
            if (static_cast<int>(v.size()) != nchan)
                throw std::invalid_argument("GRP: ragged channel dimension");
        }
    }

    std::vector<double> flat(
        static_cast<std::size_t>(nx) * ny * nchan);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int c = 0; c < nchan; ++c)
                flat[(static_cast<std::size_t>(i) * ny + j) * nchan + c]
                    = vector_field[i][j][c];
    compute_tensor_vector(flat.data(), nchan);
}

double GeneralizedRecurrencePlot::point_distance(double a, double b) const {
    // All three norms reduce to |a - b| for scalar values.
    return std::abs(a - b);
}

double GeneralizedRecurrencePlot::vector_distance(
    const std::vector<double>& a, const std::vector<double>& b) const
{
    std::size_t m = a.size();
    if (norm == "manhattan") {
        double s = 0.0;
        for (std::size_t k = 0; k < m; ++k) s += std::abs(a[k] - b[k]);
        return s;
    }
    if (norm == "supremum") {
        double s = 0.0;
        for (std::size_t k = 0; k < m; ++k) {
            double d = std::abs(a[k] - b[k]);
            if (d > s) s = d;
        }
        return s;
    }
    // euclidean
    double s = 0.0;
    for (std::size_t k = 0; k < m; ++k) {
        double d = a[k] - b[k];
        s += d * d;
    }
    return std::sqrt(s);
}

void GeneralizedRecurrencePlot::compute_tensor() {
    std::size_t N4 = total_entries();
    try {
        tensor.assign(N4, 0);
    } catch (const std::bad_alloc&) {
        std::cerr << "GRP: allocation of " << N4
                  << " bytes failed. Downsample your field." << std::endl;
        throw;
    }

    // Fill: R(i1,i2,j1,j2) = 1 iff |X(i1,i2) - X(j1,j2)| <= threshold
    // Parallelise over the outer (i1, i2) pair so each thread writes into
    // disjoint slabs of the flat tensor (write-safe without atomics).
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i1 = 0; i1 < nx; ++i1) {
        for (int i2 = 0; i2 < ny; ++i2) {
            double v_i = field[i1][i2];
            for (int j1 = 0; j1 < nx; ++j1) {
                for (int j2 = 0; j2 < ny; ++j2) {
                    double v_j = field[j1][j2];
                    double d = point_distance(v_i, v_j);
                    if (d <= threshold) {
                        tensor[flat_index(i1, i2, j1, j2)] = 1;
                    }
                }
            }
        }
    }
}

void GeneralizedRecurrencePlot::compute_tensor_embedded() {
    std::size_t N4 = total_entries();
    try {
        tensor.assign(N4, 0);
    } catch (const std::bad_alloc&) {
        std::cerr << "GRP: allocation of " << N4
                  << " bytes failed. Downsample your field." << std::endl;
        throw;
    }

    const int patch_len = m1 * m2;

    // Precompute patches for every valid anchor. Memory: nx*ny*m1*m2 doubles.
    // At N=64, m=3 this is ~295 KB, negligible compared to the N^4 tensor.
    std::vector<std::vector<double>> patches(
        static_cast<std::size_t>(nx) * ny,
        std::vector<double>(patch_len));

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i1 = 0; i1 < nx; ++i1) {
        for (int i2 = 0; i2 < ny; ++i2) {
            auto& p = patches[static_cast<std::size_t>(i1) * ny + i2];
            int idx = 0;
            for (int k1 = 0; k1 < m1; ++k1) {
                for (int k2 = 0; k2 < m2; ++k2) {
                    p[idx++] = field[i1 + k1 * tau1][i2 + k2 * tau2];
                }
            }
        }
    }

    // Fill tensor using vector distances.
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i1 = 0; i1 < nx; ++i1) {
        for (int i2 = 0; i2 < ny; ++i2) {
            const auto& pi = patches[static_cast<std::size_t>(i1) * ny + i2];
            for (int j1 = 0; j1 < nx; ++j1) {
                for (int j2 = 0; j2 < ny; ++j2) {
                    const auto& pj = patches[static_cast<std::size_t>(j1) * ny + j2];
                    if (vector_distance(pi, pj) <= threshold) {
                        tensor[flat_index(i1, i2, j1, j2)] = 1;
                    }
                }
            }
        }
    }
}

void GeneralizedRecurrencePlot::compute_tensor_vector(
    const double* vfield, int nchan)
{
    std::size_t N4 = total_entries();
    try {
        tensor.assign(N4, 0);
    } catch (const std::bad_alloc&) {
        std::cerr << "GRP: allocation of " << N4
                  << " bytes failed. Downsample your field." << std::endl;
        throw;
    }

    const int nc = nchan;
    const bool use_euclid = (norm == "euclidean");
    const bool use_manhattan = (norm == "manhattan");
    const double eps2 = threshold * threshold;  // euclidean comparison squared

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i1 = 0; i1 < nx; ++i1) {
        for (int i2 = 0; i2 < ny; ++i2) {
            const double* pi = vfield + (static_cast<std::size_t>(i1) * ny + i2) * nc;
            for (int j1 = 0; j1 < nx; ++j1) {
                for (int j2 = 0; j2 < ny; ++j2) {
                    const double* pj = vfield + (static_cast<std::size_t>(j1) * ny + j2) * nc;
                    bool within = false;
                    if (use_euclid) {
                        double d2 = 0.0;
                        for (int c = 0; c < nc; ++c) {
                            double dc = pi[c] - pj[c];
                            d2 += dc * dc;
                            if (d2 > eps2) break;  // early exit
                        }
                        within = (d2 <= eps2);
                    } else if (use_manhattan) {
                        double d = 0.0;
                        for (int c = 0; c < nc; ++c) {
                            d += std::abs(pi[c] - pj[c]);
                            if (d > threshold) break;
                        }
                        within = (d <= threshold);
                    } else {  // supremum
                        double d = 0.0;
                        for (int c = 0; c < nc; ++c) {
                            double dc = std::abs(pi[c] - pj[c]);
                            if (dc > d) d = dc;
                            if (d > threshold) break;
                        }
                        within = (d <= threshold);
                    }
                    if (within)
                        tensor[flat_index(i1, i2, j1, j2)] = 1;
                }
            }
        }
    }
}

std::size_t GeneralizedRecurrencePlot::total_recurrent_pairs() const {
    // long long chosen for the OpenMP reduction (std::size_t isn't universally
    // supported as a reduction type on older compilers).
    long long count = 0;
    const std::size_t n = tensor.size();
    #pragma omp parallel for reduction(+:count) schedule(static)
    for (std::size_t k = 0; k < n; ++k)
        count += tensor[k];
    return static_cast<std::size_t>(count);
}

void GeneralizedRecurrencePlot::write_slice_fixed_i(
    int i1, int i2, const std::string& filename) const
{
    if (i1 < 0 || i1 >= nx || i2 < 0 || i2 >= ny) {
        throw std::out_of_range("GRP: slice indices out of range");
    }
    std::ofstream out(filename);
    if (!out) throw std::runtime_error("GRP: cannot open " + filename);
    for (int j1 = 0; j1 < nx; ++j1) {
        for (int j2 = 0; j2 < ny; ++j2) {
            out << static_cast<int>(R(i1, i2, j1, j2));
            if (j2 + 1 < ny) out << ' ';
        }
        out << '\n';
    }
}

void GeneralizedRecurrencePlot::write_hsoi_subsection_fixed_i2_eq_j2(
    int k, const std::string& filename) const
{
    if (k < 0 || k >= ny) {
        throw std::out_of_range("GRP: HSOI index out of range");
    }
    // Output format: i1 j1 k R  (only recurrent entries for sparsity)
    std::ofstream out(filename);
    if (!out) throw std::runtime_error("GRP: cannot open " + filename);
    out << "# i1 j1 i2=j2=k R\n";
    for (int i1 = 0; i1 < nx; ++i1) {
        for (int j1 = 0; j1 < nx; ++j1) {
            int r = R(i1, k, j1, k);
            if (r) {
                out << i1 << ' ' << j1 << ' ' << k << ' ' << r << '\n';
            }
        }
    }
}

void GeneralizedRecurrencePlot::write_sparse_list(
    const std::string& filename) const
{
    std::ofstream out(filename);
    if (!out) throw std::runtime_error("GRP: cannot open " + filename);
    out << "# i1 i2 j1 j2\n";
    for (int i1 = 0; i1 < nx; ++i1)
        for (int i2 = 0; i2 < ny; ++i2)
            for (int j1 = 0; j1 < nx; ++j1)
                for (int j2 = 0; j2 < ny; ++j2)
                    if (R(i1, i2, j1, j2))
                        out << i1 << ' ' << i2 << ' '
                            << j1 << ' ' << j2 << '\n';
}

void GeneralizedRecurrencePlot::print_summary() const {
    std::size_t nrec = total_recurrent_pairs();
    std::size_t ntot = total_entries();
    double rr = static_cast<double>(nrec) / static_cast<double>(ntot);
    std::cout << "GeneralizedRecurrencePlot summary" << std::endl;
    std::cout << "  nx x ny        = " << nx << " x " << ny << std::endl;
    std::cout << "  tensor size    = " << ntot << " entries ("
              << (ntot / (1024.0 * 1024.0)) << " MB at 1 B/entry)" << std::endl;
    std::cout << "  threshold      = " << threshold << std::endl;
    std::cout << "  norm           = " << norm << std::endl;
    std::cout << "  recurrent pts  = " << nrec << std::endl;
    std::cout << "  recurrence rate RR = " << rr << std::endl;
}
