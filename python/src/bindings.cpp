// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Abraham Joseph Okayli Masaryk
/*
 * File:   bindings.cpp
 * Purpose: pybind11 bindings for the 2D generalised-recurrence-plot library.
 *
 * The C++ side uses std::vector<std::vector<double>> for input fields and
 * std::vector<...> for output fields. Here we accept / return numpy arrays
 * so notebooks stay idiomatic. All conversions are kept explicit and local
 * to avoid surprising performance cliffs.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "GeneralizedRecurrencePlot.h"
#include "SpatialRQA.h"
#include "SpatialRecurrenceNetwork.h"
#include "JointGeneralizedRecurrencePlot.h"
#include "CrossGeneralizedRecurrencePlot.h"
#include "SpatialSlidingWindow.h"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// NumPy <-> nested-vector conversion helpers
// ---------------------------------------------------------------------------

using Field = std::vector<std::vector<double>>;

static Field numpy_to_field(py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
    if (arr.ndim() != 2)
        throw std::invalid_argument("field must be a 2-D numpy array");
    int nx = static_cast<int>(arr.shape(0));
    int ny = static_cast<int>(arr.shape(1));
    Field f(nx, std::vector<double>(ny));
    auto r = arr.unchecked<2>();
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            f[i][j] = r(i, j);
    return f;
}

// Copy a flat, row-major vector<double> of size (nx*ny) into a 2D numpy array.
template <typename T>
static py::array_t<T> flat_to_numpy(const std::vector<T>& v, int nx, int ny) {
    if (static_cast<int>(v.size()) != nx * ny)
        throw std::runtime_error("flat_to_numpy: size mismatch");
    py::array_t<T> out({nx, ny});
    auto r = out.template mutable_unchecked<2>();
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            r(i, j) = v[static_cast<std::size_t>(i) * ny + j];
    return out;
}

// Copy a vector<vector<double>> into a 2D numpy array.
static py::array_t<double> nested_to_numpy(
    const std::vector<std::vector<double>>& v)
{
    int n1 = static_cast<int>(v.size());
    int n2 = n1 > 0 ? static_cast<int>(v[0].size()) : 0;
    py::array_t<double> out({n1, n2});
    auto r = out.mutable_unchecked<2>();
    for (int i = 0; i < n1; ++i)
        for (int j = 0; j < n2; ++j)
            r(i, j) = v[i][j];
    return out;
}
static py::array_t<int> nested_to_numpy_i(
    const std::vector<std::vector<int>>& v)
{
    int n1 = static_cast<int>(v.size());
    int n2 = n1 > 0 ? static_cast<int>(v[0].size()) : 0;
    py::array_t<int> out({n1, n2});
    auto r = out.mutable_unchecked<2>();
    for (int i = 0; i < n1; ++i)
        for (int j = 0; j < n2; ++j)
            r(i, j) = v[i][j];
    return out;
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(_core, m) {
    m.doc() = "2D generalised recurrence plots (Marwan 2007b), "
              "recurrence networks (Donner 2010), joint / cross RP, "
              "sliding windows, optional patch embedding. "
              "OpenMP-parallelised C++ backend.";

    // ------------------------------------------------------------
    // GeneralizedRecurrencePlot
    // ------------------------------------------------------------
    py::class_<GeneralizedRecurrencePlot>(m, "GeneralizedRecurrencePlot",
        "4D recurrence tensor R(i1,i2,j1,j2) = Θ(ε − |X(i1,i2) − X(j1,j2)|).")
        .def(py::init([](py::array_t<double, py::array::c_style | py::array::forcecast> field,
                         double eps, const std::string& norm) {
                // Zero-copy path: take the numpy buffer directly through the
                // flat-pointer C++ constructor. forcecast ensures C-contiguous
                // double storage (if the user passes float32 or F-order,
                // pybind11 copies once; otherwise no copy).
                if (field.ndim() != 2)
                    throw std::invalid_argument("field must be 2-D");
                auto buf = field.request();
                return new GeneralizedRecurrencePlot(
                    static_cast<const double*>(buf.ptr),
                    static_cast<int>(buf.shape[0]),
                    static_cast<int>(buf.shape[1]),
                    eps, norm);
             }),
             py::arg("field"), py::arg("threshold"),
             py::arg("norm") = "euclidean",
             "Scalar-value constructor. Zero-copy for C-contiguous float64 arrays.")

        .def(py::init([](py::array_t<double> field,
                         int m1, int m2, int tau1, int tau2,
                         double eps, const std::string& norm) {
                return new GeneralizedRecurrencePlot(
                    numpy_to_field(field),
                    m1, m2, tau1, tau2, eps, norm);
             }),
             py::arg("field"),
             py::arg("m1"), py::arg("m2"),
             py::arg("tau1"), py::arg("tau2"),
             py::arg("threshold"),
             py::arg("norm") = "euclidean",
             "Patch-embedding constructor (m1 x m2 patch, delays tau1, tau2).")


        .def_property_readonly("nx", &GeneralizedRecurrencePlot::get_nx)
        .def_property_readonly("ny", &GeneralizedRecurrencePlot::get_ny)
        .def_property_readonly("threshold",
                               &GeneralizedRecurrencePlot::get_threshold)
        .def_property_readonly("norm", &GeneralizedRecurrencePlot::get_norm)
        .def_property_readonly("total_entries",
                               &GeneralizedRecurrencePlot::total_entries)
        .def_property_readonly("total_recurrent_pairs",
                               &GeneralizedRecurrencePlot::total_recurrent_pairs)
        .def_property_readonly("RR", [](const GeneralizedRecurrencePlot& g) {
            auto ntot = g.total_entries();
            return ntot > 0
                   ? static_cast<double>(g.total_recurrent_pairs()) / ntot
                   : 0.0;
        }, "Recurrence rate = recurrent_pairs / total_entries.")

        .def("R", &GeneralizedRecurrencePlot::R,
             py::arg("i1"), py::arg("i2"), py::arg("j1"), py::arg("j2"),
             "Tensor value at (i1, i2, j1, j2); 0 or 1.")

        .def("print_summary", &GeneralizedRecurrencePlot::print_summary)

        // Vector-valued factory. Kept as a static method because pybind11
        // cannot disambiguate 2D vs 3D numpy arguments through overloaded
        // init() alone (exceptions inside the lambda propagate rather than
        // triggering overload fallback).
        .def_static("from_vector_field",
            [](py::array_t<double, py::array::c_style | py::array::forcecast> vfield,
               double eps, const std::string& norm) {
                if (vfield.ndim() != 3)
                    throw std::invalid_argument("vector field must be 3-D (nx, ny, nchan)");
                auto buf = vfield.request();
                return new GeneralizedRecurrencePlot(
                    static_cast<const double*>(buf.ptr),
                    static_cast<int>(buf.shape[0]),
                    static_cast<int>(buf.shape[1]),
                    static_cast<int>(buf.shape[2]),
                    eps, norm);
            },
            py::arg("vector_field"), py::arg("threshold"),
            py::arg("norm") = "euclidean",
            py::return_value_policy::take_ownership,
            "Build a GRP from a vector-valued field of shape (nx, ny, nchan). "
            "Each pixel is a state vector; recurrence uses ||x - y|| under the "
            "chosen norm. Breaks the 1D-scalar clustering saturation.")

        .def("slice", [](const GeneralizedRecurrencePlot& g, int i1, int i2) {
            // Return the 2D binary slice R(i1, i2, :, :) as a numpy array.
            int nx = g.get_nx(), ny = g.get_ny();
            py::array_t<std::uint8_t> out({nx, ny});
            auto r = out.mutable_unchecked<2>();
            for (int j1 = 0; j1 < nx; ++j1)
                for (int j2 = 0; j2 < ny; ++j2)
                    r(j1, j2) = static_cast<std::uint8_t>(g.R(i1, i2, j1, j2));
            return out;
        }, py::arg("i1"), py::arg("i2"),
           "2D binary slice of the tensor at fixed reference pixel (i1, i2).");

    // ------------------------------------------------------------
    // SpatialRQA
    // ------------------------------------------------------------
    py::class_<SpatialRQA>(m, "SpatialRQA",
        "Marwan 2007b generalised RQA on a GRP tensor.")
        .def(py::init<const GeneralizedRecurrencePlot&, int, int>(),
             py::arg("grp"), py::arg("l_min"), py::arg("v_min"),
             py::keep_alive<1, 2>())
        .def("calculate", &SpatialRQA::calculate,
             py::arg("l_min"), py::arg("v_min"))
        .def_property_readonly("RR", &SpatialRQA::get_RR)
        .def_property_readonly("DET_HS", &SpatialRQA::get_DET_HS)
        .def_property_readonly("LAM_HS", &SpatialRQA::get_LAM_HS)
        .def_property_readonly("L_HS", &SpatialRQA::get_L_HS)
        .def_property_readonly("TT_HS", &SpatialRQA::get_TT_HS)
        .def_property_readonly("Pl", [](const SpatialRQA& r) {
            auto& p = r.get_Pl();
            py::array_t<long long> out(p.size());
            auto a = out.mutable_unchecked<1>();
            for (std::size_t i = 0; i < p.size(); ++i) a(i) = p[i];
            return out;
        })
        .def_property_readonly("Pv", [](const SpatialRQA& r) {
            auto& p = r.get_Pv();
            py::array_t<long long> out(p.size());
            auto a = out.mutable_unchecked<1>();
            for (std::size_t i = 0; i < p.size(); ++i) a(i) = p[i];
            return out;
        })
        .def_property_readonly("l_min", &SpatialRQA::get_l_min)
        .def_property_readonly("v_min", &SpatialRQA::get_v_min)
        .def("print_measures", &SpatialRQA::print_measures)
        .def("measures", [](const SpatialRQA& r) {
            py::dict d;
            d["RR"]     = r.get_RR();
            d["DET_HS"] = r.get_DET_HS();
            d["LAM_HS"] = r.get_LAM_HS();
            d["L_HS"]   = r.get_L_HS();
            d["TT_HS"]  = r.get_TT_HS();
            d["l_min"]  = r.get_l_min();
            d["v_min"]  = r.get_v_min();
            return d;
        }, "Return all scalar measures as a dict.");

    // ------------------------------------------------------------
    // SpatialRecurrenceNetwork
    // ------------------------------------------------------------
    py::class_<SpatialRecurrenceNetwork>(m, "SpatialRecurrenceNetwork",
        "Recurrence-network characterisation of a GRP tensor (Donner 2010).")
        .def(py::init<const GeneralizedRecurrencePlot&>(),
             py::arg("grp"),
             py::keep_alive<1, 2>())
        .def("compute_assortativity",
             &SpatialRecurrenceNetwork::compute_assortativity)
        .def("compute_closeness",
             &SpatialRecurrenceNetwork::compute_closeness)
        .def("compute_betweenness",
             &SpatialRecurrenceNetwork::compute_betweenness)
        .def("matching_index",
             &SpatialRecurrenceNetwork::matching_index,
             py::arg("u"), py::arg("v"))
        .def("compute_matching_field",
             [](SpatialRecurrenceNetwork& s) {
                const auto& v = s.compute_matching_field();
                int N = s.get_N();
                py::array_t<float> out({N, N});
                auto r = out.mutable_unchecked<2>();
                for (int i = 0; i < N; ++i)
                    for (int j = 0; j < N; ++j)
                        r(i, j) = v[static_cast<std::size_t>(i) * N + j];
                return out;
             },
             "Full N*N matching-index matrix (float32). Heavy!")

        // Per-vertex fields reshaped back to (nx, ny)
        .def_property_readonly("degree", [](const SpatialRecurrenceNetwork& s) {
            return flat_to_numpy<long long>(
                s.get_degree(), s.get_nx(), s.get_ny());
        })
        .def_property_readonly("clustering", [](const SpatialRecurrenceNetwork& s) {
            return flat_to_numpy<double>(
                s.get_clustering(), s.get_nx(), s.get_ny());
        })
        .def_property_readonly("closeness", [](const SpatialRecurrenceNetwork& s) {
            return flat_to_numpy<double>(
                s.get_closeness(), s.get_nx(), s.get_ny());
        })
        .def_property_readonly("betweenness", [](const SpatialRecurrenceNetwork& s) {
            return flat_to_numpy<double>(
                s.get_betweenness(), s.get_nx(), s.get_ny());
        })

        .def_property_readonly("mean_degree",
            &SpatialRecurrenceNetwork::get_mean_degree)
        .def_property_readonly("mean_clustering",
            &SpatialRecurrenceNetwork::get_mean_clustering)
        .def_property_readonly("transitivity",
            &SpatialRecurrenceNetwork::get_transitivity)
        .def_property_readonly("assortativity",
            &SpatialRecurrenceNetwork::get_assortativity)
        .def_property_readonly("edge_density",
            &SpatialRecurrenceNetwork::get_edge_density)
        .def_property_readonly("nx", &SpatialRecurrenceNetwork::get_nx)
        .def_property_readonly("ny", &SpatialRecurrenceNetwork::get_ny)
        .def_property_readonly("N", &SpatialRecurrenceNetwork::get_N)
        .def("print_summary", &SpatialRecurrenceNetwork::print_summary);

    // ------------------------------------------------------------
    // JointGeneralizedRecurrencePlot
    // ------------------------------------------------------------
    py::class_<JointGeneralizedRecurrencePlot>(m, "JointGeneralizedRecurrencePlot",
        "Joint 2D recurrence: R^JRP = R^X * R^Y on co-registered fields.")
        .def(py::init([](py::array_t<double> X, double eps_X,
                          py::array_t<double> Y, double eps_Y,
                          const std::string& norm) {
                return new JointGeneralizedRecurrencePlot(
                    numpy_to_field(X), eps_X,
                    numpy_to_field(Y), eps_Y, norm);
             }),
             py::arg("X"), py::arg("eps_X"),
             py::arg("Y"), py::arg("eps_Y"),
             py::arg("norm") = "euclidean")

        .def_property_readonly("grp",
            &JointGeneralizedRecurrencePlot::grp,
            py::return_value_policy::reference_internal,
            "GeneralizedRecurrencePlot facade exposing the joint tensor.")
        .def_property_readonly("eps_X",
            &JointGeneralizedRecurrencePlot::get_eps_X)
        .def_property_readonly("eps_Y",
            &JointGeneralizedRecurrencePlot::get_eps_Y)
        .def("print_summary",
             &JointGeneralizedRecurrencePlot::print_summary);

    // ------------------------------------------------------------
    // CrossGeneralizedRecurrencePlot
    // ------------------------------------------------------------
    py::class_<CrossGeneralizedRecurrencePlot>(m, "CrossGeneralizedRecurrencePlot",
        "Asymmetric 2D cross-recurrence tensor between two fields.")
        .def(py::init([](py::array_t<double> X, py::array_t<double> Y,
                          double eps, const std::string& norm) {
                return new CrossGeneralizedRecurrencePlot(
                    numpy_to_field(X), numpy_to_field(Y), eps, norm);
             }),
             py::arg("X"), py::arg("Y"),
             py::arg("threshold"),
             py::arg("norm") = "euclidean")

        .def("compute_cross_degrees",
            &CrossGeneralizedRecurrencePlot::compute_cross_degrees)
        .def_property_readonly("CRR",
            &CrossGeneralizedRecurrencePlot::CRR)
        .def_property_readonly("total_entries",
            &CrossGeneralizedRecurrencePlot::total_entries)
        .def_property_readonly("cross_degree_X",
            [](const CrossGeneralizedRecurrencePlot& c) {
                return flat_to_numpy<long long>(
                    c.get_cross_degree_X(), c.get_nx1(), c.get_ny1());
            })
        .def_property_readonly("cross_degree_Y",
            [](const CrossGeneralizedRecurrencePlot& c) {
                return flat_to_numpy<long long>(
                    c.get_cross_degree_Y(), c.get_nx2(), c.get_ny2());
            })
        .def("xmatch_XX",
            &CrossGeneralizedRecurrencePlot::xmatch_XX,
            py::arg("u1"), py::arg("u2"))
        .def("xmatch_YY",
            &CrossGeneralizedRecurrencePlot::xmatch_YY,
            py::arg("v1"), py::arg("v2"))
        .def_property_readonly("nx1",
            &CrossGeneralizedRecurrencePlot::get_nx1)
        .def_property_readonly("ny1",
            &CrossGeneralizedRecurrencePlot::get_ny1)
        .def_property_readonly("nx2",
            &CrossGeneralizedRecurrencePlot::get_nx2)
        .def_property_readonly("ny2",
            &CrossGeneralizedRecurrencePlot::get_ny2)
        .def_property_readonly("NX",
            &CrossGeneralizedRecurrencePlot::get_NX)
        .def_property_readonly("NY",
            &CrossGeneralizedRecurrencePlot::get_NY)
        .def("print_summary",
             &CrossGeneralizedRecurrencePlot::print_summary);

    // ------------------------------------------------------------
    // SpatialSlidingWindow
    // ------------------------------------------------------------
    // The struct Result is also exposed so users can poke around.
    py::class_<SpatialSlidingWindow::Result>(m, "SlidingWindowResult")
        .def_readonly("wx",           &SpatialSlidingWindow::Result::wx)
        .def_readonly("wy",           &SpatialSlidingWindow::Result::wy)
        .def_readonly("stride_x",     &SpatialSlidingWindow::Result::stride_x)
        .def_readonly("stride_y",     &SpatialSlidingWindow::Result::stride_y)
        .def_readonly("n_windows_x",  &SpatialSlidingWindow::Result::n_windows_x)
        .def_readonly("n_windows_y",  &SpatialSlidingWindow::Result::n_windows_y)
        .def_property_readonly("origin_x",        [](const SpatialSlidingWindow::Result& r){ return nested_to_numpy_i(r.origin_x); })
        .def_property_readonly("origin_y",        [](const SpatialSlidingWindow::Result& r){ return nested_to_numpy_i(r.origin_y); })
        .def_property_readonly("RR",              [](const SpatialSlidingWindow::Result& r){ return nested_to_numpy(r.RR); })
        .def_property_readonly("DET_HS",          [](const SpatialSlidingWindow::Result& r){ return nested_to_numpy(r.DET_HS); })
        .def_property_readonly("LAM_HS",          [](const SpatialSlidingWindow::Result& r){ return nested_to_numpy(r.LAM_HS); })
        .def_property_readonly("L_HS",            [](const SpatialSlidingWindow::Result& r){ return nested_to_numpy(r.L_HS); })
        .def_property_readonly("TT_HS",           [](const SpatialSlidingWindow::Result& r){ return nested_to_numpy(r.TT_HS); })
        .def_property_readonly("mean_degree",     [](const SpatialSlidingWindow::Result& r){ return nested_to_numpy(r.mean_degree); })
        .def_property_readonly("mean_clustering", [](const SpatialSlidingWindow::Result& r){ return nested_to_numpy(r.mean_clustering); })
        .def_property_readonly("transitivity",    [](const SpatialSlidingWindow::Result& r){ return nested_to_numpy(r.transitivity); })
        .def_property_readonly("assortativity",   [](const SpatialSlidingWindow::Result& r){ return nested_to_numpy(r.assortativity); })
        .def_property_readonly("eps_used",        [](const SpatialSlidingWindow::Result& r){ return nested_to_numpy(r.eps_used); });

    m.def("sliding_window_fixed_eps",
          [](py::array_t<double> field, int wx, int wy,
             int stride_x, int stride_y, double eps,
             int l_min, int v_min, bool compute_network,
             const std::string& norm) {
              return SpatialSlidingWindow::run_fixed_eps(
                  numpy_to_field(field), wx, wy,
                  stride_x, stride_y, eps, l_min, v_min,
                  compute_network, norm);
          },
          py::arg("field"), py::arg("wx"), py::arg("wy"),
          py::arg("stride_x"), py::arg("stride_y"),
          py::arg("eps"),
          py::arg("l_min") = 2, py::arg("v_min") = 2,
          py::arg("compute_network") = true,
          py::arg("norm") = "euclidean",
          "Sliding-window GRP + SpatialRQA + SRN with a fixed threshold.");

    m.def("sliding_window_target_rr",
          [](py::array_t<double> field, int wx, int wy,
             int stride_x, int stride_y, double target_rr,
             int l_min, int v_min, bool compute_network,
             const std::string& norm, unsigned seed) {
              return SpatialSlidingWindow::run_target_rr(
                  numpy_to_field(field), wx, wy,
                  stride_x, stride_y, target_rr, l_min, v_min,
                  compute_network, norm, seed);
          },
          py::arg("field"), py::arg("wx"), py::arg("wy"),
          py::arg("stride_x"), py::arg("stride_y"),
          py::arg("target_rr"),
          py::arg("l_min") = 2, py::arg("v_min") = 2,
          py::arg("compute_network") = true,
          py::arg("norm") = "euclidean", py::arg("seed") = 42u,
          "Per-window adaptive threshold to hit target_rr.");
}
