// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Abraham Joseph Okayli Masaryk
/*
 * File:   SpatialRecurrenceNetwork.cpp
 */

#include "SpatialRecurrenceNetwork.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <queue>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

SpatialRecurrenceNetwork::SpatialRecurrenceNetwork(
    const GeneralizedRecurrencePlot& grp_in)
    : grp(grp_in),
      nx(grp_in.get_nx()),
      ny(grp_in.get_ny()),
      N(grp_in.get_nx() * grp_in.get_ny()),
      nbrs_built(false),
      mean_k(0.0), mean_C(0.0),
      transitivity(0.0), assortativity(0.0), edge_density(0.0)
{
    compute_basic_measures();
}

// ----------------------------------------------------------------------
// Neighbour list (lazy)
// ----------------------------------------------------------------------
void SpatialRecurrenceNetwork::build_neighbours() {
    if (nbrs_built) return;
    nbrs.assign(N, {});

    // Each vertex gets its row populated by a single thread.
    #pragma omp parallel for schedule(dynamic)
    for (int u = 0; u < N; ++u) {
        int i1, i2;
        ij_of_vertex(u, i1, i2);
        std::vector<int>& row = nbrs[u];
        row.reserve(64);
        for (int v = 0; v < N; ++v) {
            if (u == v) continue;
            int j1, j2;
            ij_of_vertex(v, j1, j2);
            if (grp.R(i1, i2, j1, j2)) row.push_back(v);
        }
    }
    nbrs_built = true;
}

const std::vector<std::vector<int>>& SpatialRecurrenceNetwork::get_neighbours() {
    if (!nbrs_built) build_neighbours();
    return nbrs;
}

// ----------------------------------------------------------------------
// Basic measures: degree, local clustering, global aggregates
// ----------------------------------------------------------------------
void SpatialRecurrenceNetwork::compute_basic_measures() {
    build_neighbours();

    degree.assign(N, 0);
    clustering.assign(N, 0.0);

    long long total_triangles_x6 = 0;  // each triangle counted 6x (3 vertices * 2 orderings)
    long long total_triples      = 0;  // ordered pairs of neighbours

    #pragma omp parallel for reduction(+:total_triangles_x6, total_triples) \
                             schedule(dynamic)
    for (int u = 0; u < N; ++u) {
        const auto& Nu = nbrs[u];
        const int k = static_cast<int>(Nu.size());
        degree[u] = k;

        if (k < 2) { clustering[u] = 0.0; continue; }

        // Count connected pairs of neighbours (= triangles through u).
        long long triangles_u = 0;
        for (int a = 0; a < k; ++a) {
            int va = Nu[a];
            int i1a, i2a; ij_of_vertex(va, i1a, i2a);
            for (int b = a + 1; b < k; ++b) {
                int vb = Nu[b];
                int j1b, j2b; ij_of_vertex(vb, j1b, j2b);
                if (grp.R(i1a, i2a, j1b, j2b)) ++triangles_u;
            }
        }

        double denom = static_cast<double>(k) * (k - 1) / 2.0;
        clustering[u] = (denom > 0) ? (triangles_u / denom) : 0.0;

        // Each triangle through u is counted once here, but each triangle
        // appears at all 3 vertices — so total across vertices = 3x triangles.
        // Multiply by 2 for the 6x factor used in transitivity below.
        total_triangles_x6 += 2 * triangles_u;
        total_triples      += static_cast<long long>(k) * (k - 1) / 2;
    }

    // Aggregates
    long long sum_k = 0;
    for (int u = 0; u < N; ++u) sum_k += degree[u];
    mean_k = (N > 0) ? static_cast<double>(sum_k) / N : 0.0;

    double sum_C = 0.0;
    for (int u = 0; u < N; ++u) sum_C += clustering[u];
    mean_C = (N > 0) ? sum_C / N : 0.0;

    // Transitivity = 3 * triangles / triples; triangles = total_triangles_x6 / 6
    // Triples (open+closed, unordered) = total_triples; so
    //   transitivity = (total_triangles_x6 / 2) / total_triples
    transitivity = (total_triples > 0)
                 ? (static_cast<double>(total_triangles_x6) / 2.0)
                   / static_cast<double>(total_triples)
                 : 0.0;

    // Edge density = sum_k / (N * (N - 1))
    edge_density = (N > 1)
                 ? static_cast<double>(sum_k) / (static_cast<double>(N) * (N - 1))
                 : 0.0;
}

// ----------------------------------------------------------------------
// Assortativity (Newman)
//   r = [ L^-1 sum_{ij in E} k_i k_j  -  (L^-1 sum_{ij in E} 0.5(k_i + k_j))^2 ]
//       / [ L^-1 sum_{ij in E} 0.5(k_i^2 + k_j^2)  -  (same)^2 ]
// ----------------------------------------------------------------------
void SpatialRecurrenceNetwork::compute_assortativity() {
    build_neighbours();

    // Work only on the half-edges (u < v) to avoid double counting.
    long long n_edges = 0;
    double s1 = 0.0;  // sum of k_i * k_j
    double s2 = 0.0;  // sum of 0.5 * (k_i + k_j)
    double s3 = 0.0;  // sum of 0.5 * (k_i^2 + k_j^2)

    #pragma omp parallel for reduction(+:n_edges, s1, s2, s3) schedule(dynamic)
    for (int u = 0; u < N; ++u) {
        const auto& Nu = nbrs[u];
        long long ku = degree[u];
        for (int v : Nu) {
            if (v <= u) continue;
            long long kv = degree[v];
            ++n_edges;
            s1 += static_cast<double>(ku) * kv;
            s2 += 0.5 * (ku + kv);
            s3 += 0.5 * (static_cast<double>(ku) * ku
                       + static_cast<double>(kv) * kv);
        }
    }

    if (n_edges == 0) { assortativity = 0.0; return; }
    double invL = 1.0 / static_cast<double>(n_edges);
    double num = invL * s1 - (invL * s2) * (invL * s2);
    double den = invL * s3 - (invL * s2) * (invL * s2);
    assortativity = (std::abs(den) > 1e-15) ? (num / den) : 0.0;
}

// ----------------------------------------------------------------------
// Closeness centrality (BFS from each source)
//   C_v = (N - 1) / sum_{u != v} d(u, v)     (unreachable: infinity -> 0)
// We compute the mean over connected vertices only (harmonic form),
// which is more informative for fragmented networks.
// ----------------------------------------------------------------------
void SpatialRecurrenceNetwork::compute_closeness() {
    build_neighbours();
    closeness.assign(N, 0.0);

    #pragma omp parallel
    {
        std::vector<int> dist(N);
        std::queue<int> q;

        #pragma omp for schedule(dynamic)
        for (int s = 0; s < N; ++s) {
            std::fill(dist.begin(), dist.end(), -1);
            dist[s] = 0;
            q = std::queue<int>();
            q.push(s);
            long long sum_d = 0;
            int n_reached = 0;
            while (!q.empty()) {
                int u = q.front(); q.pop();
                for (int v : nbrs[u]) {
                    if (dist[v] < 0) {
                        dist[v] = dist[u] + 1;
                        sum_d += dist[v];
                        ++n_reached;
                        q.push(v);
                    }
                }
            }
            closeness[s] = (sum_d > 0)
                         ? static_cast<double>(n_reached) / sum_d
                         : 0.0;
        }
    }
}

// ----------------------------------------------------------------------
// Betweenness centrality (Brandes 2001).
// Unweighted, undirected. One BFS per source accumulates dependencies.
// ----------------------------------------------------------------------
void SpatialRecurrenceNetwork::compute_betweenness() {
    build_neighbours();
    betweenness.assign(N, 0.0);

    // Thread-local partials to avoid atomics in the inner loop.
    #pragma omp parallel
    {
        std::vector<double> bc_local(N, 0.0);
        std::vector<double> sigma(N);
        std::vector<int>    dist(N);
        std::vector<std::vector<int>> preds(N);
        std::vector<double> delta(N);
        std::vector<int>    order;
        order.reserve(N);

        #pragma omp for schedule(dynamic)
        for (int s = 0; s < N; ++s) {
            // Init
            std::fill(sigma.begin(), sigma.end(), 0.0);
            std::fill(dist.begin(),  dist.end(),  -1);
            std::fill(delta.begin(), delta.end(), 0.0);
            for (auto& p : preds) p.clear();
            order.clear();

            sigma[s] = 1.0;
            dist[s]  = 0;
            std::queue<int> q;
            q.push(s);
            while (!q.empty()) {
                int u = q.front(); q.pop();
                order.push_back(u);
                for (int v : nbrs[u]) {
                    if (dist[v] < 0) {
                        dist[v] = dist[u] + 1;
                        q.push(v);
                    }
                    if (dist[v] == dist[u] + 1) {
                        sigma[v] += sigma[u];
                        preds[v].push_back(u);
                    }
                }
            }
            // Accumulate dependencies in reverse BFS order.
            for (auto it = order.rbegin(); it != order.rend(); ++it) {
                int w = *it;
                for (int u : preds[w]) {
                    double frac = (sigma[w] > 0)
                                ? (sigma[u] / sigma[w]) * (1.0 + delta[w])
                                : 0.0;
                    delta[u] += frac;
                }
                if (w != s) bc_local[w] += delta[w];
            }
        }

        #pragma omp critical
        for (int u = 0; u < N; ++u) betweenness[u] += bc_local[u];
    }

    // Undirected graph: each pair's contribution counted twice.
    for (int u = 0; u < N; ++u) betweenness[u] *= 0.5;
}

// ----------------------------------------------------------------------
// Matching index
//   mu(u, v) = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|   (Jaccard of neighbourhoods)
// ----------------------------------------------------------------------
double SpatialRecurrenceNetwork::matching_index(int u, int v) const {
    if (u == v) return 1.0;
    if (!nbrs_built) {
        const_cast<SpatialRecurrenceNetwork*>(this)->build_neighbours();
    }
    const auto& Nu = nbrs[u];
    const auto& Nv = nbrs[v];
    // Both lists are sorted (filled by ascending v).
    int i = 0, j = 0;
    int inter = 0;
    while (i < (int)Nu.size() && j < (int)Nv.size()) {
        if      (Nu[i] < Nv[j]) ++i;
        else if (Nu[i] > Nv[j]) ++j;
        else { ++inter; ++i; ++j; }
    }
    int uni = static_cast<int>(Nu.size() + Nv.size()) - inter;
    return (uni > 0) ? static_cast<double>(inter) / uni : 0.0;
}

const std::vector<float>& SpatialRecurrenceNetwork::compute_matching_field() {
    build_neighbours();
    std::size_t total = static_cast<std::size_t>(N) * N;
    matching_field.assign(total, 0.0f);

    #pragma omp parallel for schedule(dynamic)
    for (int u = 0; u < N; ++u) {
        const auto& Nu = nbrs[u];
        for (int v = u + 1; v < N; ++v) {
            const auto& Nv = nbrs[v];
            int i = 0, j = 0, inter = 0;
            while (i < (int)Nu.size() && j < (int)Nv.size()) {
                if      (Nu[i] < Nv[j]) ++i;
                else if (Nu[i] > Nv[j]) ++j;
                else { ++inter; ++i; ++j; }
            }
            int uni = static_cast<int>(Nu.size() + Nv.size()) - inter;
            float mu = (uni > 0) ? static_cast<float>(inter) / uni : 0.0f;
            matching_field[static_cast<std::size_t>(u) * N + v] = mu;
            matching_field[static_cast<std::size_t>(v) * N + u] = mu;
        }
        matching_field[static_cast<std::size_t>(u) * N + u] = 1.0f;
    }
    return matching_field;
}

// ----------------------------------------------------------------------
// I/O
// ----------------------------------------------------------------------
void SpatialRecurrenceNetwork::print_summary() const {
    std::cout << "SpatialRecurrenceNetwork summary" << std::endl;
    std::cout << "  N (vertices)         = " << N << std::endl;
    std::cout << "  edge density         = " << edge_density << std::endl;
    std::cout << "  mean degree          = " << mean_k << std::endl;
    std::cout << "  mean clustering      = " << mean_C << std::endl;
    std::cout << "  transitivity         = " << transitivity << std::endl;
    if (assortativity != 0.0) {
        std::cout << "  assortativity        = " << assortativity << std::endl;
    }
}

void SpatialRecurrenceNetwork::write_field(const std::vector<double>& f,
                                           const std::string& filename) const
{
    std::ofstream out(filename);
    if (!out) throw std::runtime_error("SRN: cannot open " + filename);
    for (int i1 = 0; i1 < nx; ++i1) {
        for (int i2 = 0; i2 < ny; ++i2) {
            out << f[vertex_of(i1, i2)];
            if (i2 + 1 < ny) out << ' ';
        }
        out << '\n';
    }
}

void SpatialRecurrenceNetwork::write_field(const std::vector<long long>& f,
                                           const std::string& filename) const
{
    std::ofstream out(filename);
    if (!out) throw std::runtime_error("SRN: cannot open " + filename);
    for (int i1 = 0; i1 < nx; ++i1) {
        for (int i2 = 0; i2 < ny; ++i2) {
            out << f[vertex_of(i1, i2)];
            if (i2 + 1 < ny) out << ' ';
        }
        out << '\n';
    }
}

void SpatialRecurrenceNetwork::write_degree_field(const std::string& f) const {
    write_field(degree, f);
}
void SpatialRecurrenceNetwork::write_clustering_field(const std::string& f) const {
    write_field(clustering, f);
}
void SpatialRecurrenceNetwork::write_closeness_field(const std::string& f) const {
    write_field(closeness, f);
}
void SpatialRecurrenceNetwork::write_betweenness_field(const std::string& f) const {
    write_field(betweenness, f);
}
