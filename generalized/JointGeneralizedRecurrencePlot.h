// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Abraham Joseph Okayli Masaryk
/*
 * File:   JointGeneralizedRecurrencePlot.h
 * Purpose: Joint 2D generalised recurrence plot for two co-registered scalar
 *          fields X, Y on the same nx x ny grid (Marwan/Romano/Thiel/Kurths
 *          2007 joint-recurrence extended to 2D via Marwan 2007b).
 *
 * Definition:
 *   R^JRP(i1,i2,j1,j2) = R^X(i1,i2,j1,j2) * R^Y(i1,i2,j1,j2)
 *
 * where R^X, R^Y are the individual generalised recurrence tensors with
 * per-field thresholds eps_X, eps_Y. Pixel pairs are linked iff BOTH fields
 * recur simultaneously. The resulting tensor is shape-compatible with
 * GeneralizedRecurrencePlot, so SpatialRQA and SpatialRecurrenceNetwork
 * apply unchanged.
 *
 * Typical use:
 *   JointGeneralizedRecurrencePlot J(X, eps_X, Y, eps_Y);
 *   SpatialRQA rqa(J.grp(), l_min, v_min);
 *   SpatialRecurrenceNetwork srn(J.grp());
 */

#ifndef JOINTGENERALIZEDRECURRENCEPLOT_H
#define JOINTGENERALIZEDRECURRENCEPLOT_H

#include "GeneralizedRecurrencePlot.h"

#include <string>
#include <vector>

class JointGeneralizedRecurrencePlot {
public:
    JointGeneralizedRecurrencePlot(
        const std::vector<std::vector<double>>& X, double eps_X,
        const std::vector<std::vector<double>>& Y, double eps_Y,
        const std::string& norm = "euclidean");

    // Access the combined tensor through a GRP facade (for SpatialRQA etc.).
    const GeneralizedRecurrencePlot& grp() const { return combined; }

    double get_eps_X() const { return eps_X; }
    double get_eps_Y() const { return eps_Y; }

    void print_summary() const;

private:
    double eps_X;
    double eps_Y;
    GeneralizedRecurrencePlot combined;  // holds the AND'd tensor

    // Build the joint tensor by AND-ing per-pixel two field recurrences.
    // Validates shape and parameters; throws on mismatch.
    static std::vector<std::uint8_t> build_joint_tensor(
        const std::vector<std::vector<double>>& X, double eps_X,
        const std::vector<std::vector<double>>& Y, double eps_Y,
        const std::string& norm);
};

#endif /* JOINTGENERALIZEDRECURRENCEPLOT_H */
