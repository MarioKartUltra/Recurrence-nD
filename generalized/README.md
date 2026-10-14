# Generalised Recurrence Plot (Marwan 2007b)

C++ implementation of the spatial 2D extension of recurrence plots and
recurrence quantification analysis from:

> Marwan, N., Kurths, J., & Saparin, P. (2007).
> *Generalised recurrence plot analysis for spatial data.*
> Physics Letters A, 360(4–5), 545–551.

## What this does

For a 2D scalar field `X(i1, i2)` on an `nx x ny` grid, it computes the
4D recurrence tensor

    R(i1, i2, j1, j2) = Theta( eps - |X(i1,i2) - X(j1,j2)| )

and its generalised RQA measures:

- `RR`     — recurrence rate
- `DET_HS` — determinism via diagonal hyper-surfaces (Eq. 8)
- `LAM_HS` — laminarity via vertical hyper-surfaces (Eq. 10)
- `L_HS`   — averaged diagonal HS size
- `TT_HS`  — trapping size

## Files

- `GeneralizedRecurrencePlot.h/cpp` — 4D tensor storage and I/O
- `SpatialRQA.h/cpp`                — generalised RQA measures
- `test_grp.cpp`                    — benchmark reproducing Marwan's Table 2
- `Makefile`                        — build rules

## Build and run

    make
    make run          # uses N=64 by default
    make run_small    # N=32 — very fast
    ./test_grp 128    # custom size

## Memory requirements

The 4D tensor uses `N^4` bytes. Plan accordingly:

| N   | tensor size | computation |
|-----|-------------|-------------|
|  32 |      1 MB   | seconds     |
|  64 |     16 MB   | seconds     |
| 100 |    100 MB   | ~10 seconds |
| 128 |    256 MB   | ~30 seconds |
| 200 |    1.6 GB   | ~few min    |
| 256 |    4.3 GB   | requires lots of RAM |

For very large images, downsample first or restrict to regions of interest.

## Benchmark against Marwan Table 2

At `N = 200`, `eps = 0.2`, `l_min = 3`, `v_min = 4`, Marwan et al. report:

| Case     | RR    | DET_HS | LAM_HS | L_HS | TT_HS |
|----------|-------|--------|--------|------|-------|
| noise    | 0.218 | 0.007  | 0.006  | 3.7  | 3.0   |
| 2D-AR2   | 0.221 | 0.032  | 0.065  | 3.1  | 3.1   |
| periodic | 0.219 | 0.322  | 0.312  | 5.8  | 5.6   |

At smaller N the absolute values shift, but the qualitative ordering
`noise < AR2 < periodic` for DET and LAM should hold.

## Use in geology / rock mechanics

For a 2D EBSD field (GND density, KAM, orientation components, phase
indicator) on a deformed rock section, the generalised RQA measures quantify:

- **High DET_HS** — recurrent microstructural patterns; deterministic fabric.
- **High LAM_HS** — extended laminar zones where the field is flat
  (slow-varying regions, well-mixed domains).
- **L_HS, TT_HS** — characteristic sizes of the recurrent/laminar structures
  in pixels.

Compared to traditional n-point correlation functions and WTMM:
- `RR` is closely related to the histogram of value differences.
- `DET_HS` detects deterministic structure that simple autocorrelation misses.
- `LAM_HS` captures plateau-like domains that multifractal width misses.

## Limitations of the current implementation

1. Full 4D tensor storage (1 byte per entry). Bit-packing would give 8x
   reduction.
2. Diagonal and vertical HS scans are O(N^4 * l_max^2) worst case. A
   2D maximal-square dynamic program would reduce this to O(N^4).
3. Single-threaded. The tensor build loop and scans are both naturally
   parallelisable with OpenMP; not implemented here.
4. Scalar fields only. For vector-valued inputs (e.g. full orientation,
   tensor components), extend `point_distance()` and pass vector data.

These are straightforward optimisations to add after the baseline has
been validated against Marwan Table 2.
