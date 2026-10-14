// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Abraham Joseph Okayli Masaryk
/*
 * File:   SpatialRecurrenceNetwork.h
 * Purpose: Complex-network characterisation of a 2D generalised recurrence
 *          plot, following Donner et al., New J. Phys. 12 (2010) 033025.
 *
 * The 4D tensor R(i1,i2,j1,j2) is read as the adjacency matrix of an
 * unweighted undirected graph with N = nx*ny vertices. Each pixel
 * (i1, i2) becomes vertex u = i1*ny + i2. Self-loops (u == v) are
 * excluded by convention.
 *
 * Per-vertex measures (vectors of length N, spatial fields on the nx x ny grid):
 *   degree_v      = sum_u A(u, v)
 *   clustering_v  = 2 * T_v / (k_v * (k_v - 1))         (local clustering)
 *   closeness_v   = (N - 1) / sum_u d(u, v)             (shortest-path; BFS)
 *   betweenness_v = sum_{s!=t!=v} sigma_{st}(v) / sigma_{st}  (on demand)
 *
 * Global measures:
 *   mean_degree, mean_clustering
 *   transitivity T = 3 * num_triangles / num_triples
 *   assortativity r = Pearson correlation of degrees across edges
 *
 * Edge-level measures:
 *   matching_index(u, v) = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|   (Jaccard of nbhds)
 *
 * All scans are OpenMP-parallel over the outer vertex. Clustering is
 * O(N * <k^2>) worst case; betweenness (BFS-per-source) is O(N * (N+E)).
 */

#ifndef SPATIALRECURRENCENETWORK_H
#define SPATIALRECURRENCENETWORK_H

#include "GeneralizedRecurrencePlot.h"

#include <cstddef>
#include <string>
#include <vector>

class SpatialRecurrenceNetwork {
public:
    explicit SpatialRecurrenceNetwork(const GeneralizedRecurrencePlot& grp);

    // Compute per-vertex degree and clustering + global aggregates.
    // Cheap: O(N^2) and O(N * <k^2>). Called by default.
    void compute_basic_measures();

    // Optional, more expensive.
    void compute_closeness();       // O(N * (N + E))
    void compute_betweenness();     // O(N * (N + E)), Brandes algorithm
    void compute_assortativity();   // O(E)

    // Matching index between two vertices (on demand; cheap per call).
    double matching_index(int u, int v) const;

    // Build the full N*N matching-index matrix (heavy: O(N^2 * <k>)).
    // Stored row-major of size N*N as float to cap memory.
    const std::vector<float>& compute_matching_field();
    const std::vector<float>& get_matching_field() const { return matching_field; }

    // Getters
    const std::vector<long long>& get_degree()       const { return degree; }
    const std::vector<double>&    get_clustering()   const { return clustering; }
    const std::vector<double>&    get_closeness()    const { return closeness; }
    const std::vector<double>&    get_betweenness()  const { return betweenness; }

    double get_mean_degree()     const { return mean_k; }
    double get_mean_clustering() const { return mean_C; }
    double get_transitivity()    const { return transitivity; }
    double get_assortativity()   const { return assortativity; }
    double get_edge_density()    const { return edge_density; }

    int get_nx() const { return nx; }
    int get_ny() const { return ny; }
    int get_N()  const { return N;  }

    // Sparse adjacency (built lazily on first need; used by closeness and
    // betweenness). Returns CSR-like vector-of-vectors.
    const std::vector<std::vector<int>>& get_neighbours();

    // Summary + I/O on the spatial grid.
    void print_summary() const;
    void write_degree_field(const std::string& filename) const;
    void write_clustering_field(const std::string& filename) const;
    void write_closeness_field(const std::string& filename) const;
    void write_betweenness_field(const std::string& filename) const;

private:
    const GeneralizedRecurrencePlot& grp;
    int nx;
    int ny;
    int N;              // nx * ny

    // Per-vertex fields (flat layout matching pixel (i1, i2) -> u = i1*ny+i2)
    std::vector<long long> degree;
    std::vector<double>    clustering;
    std::vector<double>    closeness;
    std::vector<double>    betweenness;

    // Lazily constructed adjacency list
    bool nbrs_built;
    std::vector<std::vector<int>> nbrs;

    // Optional dense matching field, N*N floats
    std::vector<float> matching_field;

    // Global aggregates
    double mean_k;
    double mean_C;
    double transitivity;   // global clustering coefficient
    double assortativity;  // Newman's r
    double edge_density;

    // Helpers
    inline int vertex_of(int i1, int i2) const { return i1 * ny + i2; }
    inline void ij_of_vertex(int u, int& i1, int& i2) const {
        i1 = u / ny;
        i2 = u % ny;
    }
    inline int A(int u, int v) const {
        if (u == v) return 0;
        int i1, i2, j1, j2;
        ij_of_vertex(u, i1, i2);
        ij_of_vertex(v, j1, j2);
        return grp.R(i1, i2, j1, j2);
    }
    inline int A(int i1, int i2, int j1, int j2) const {
        if (i1 == j1 && i2 == j2) return 0;
        return grp.R(i1, i2, j1, j2);
    }

    void build_neighbours();

    void write_field(const std::vector<double>& f,
                     const std::string& filename) const;
    void write_field(const std::vector<long long>& f,
                     const std::string& filename) const;
};

#endif /* SPATIALRECURRENCENETWORK_H */
