# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""EBSD loaders for Oxford Instruments .ctf (text) and .cpr/.crc (binary).

Ported from the ebsd_qtz.ipynb notebook in the wavelet project so that
recurrence-analysis notebooks can load the Kuckaus, Calamita, Pearce, and
Josephine datasets directly.

Each reader returns a dict with 2D arrays for each field and a ``meta``
sub-dict describing grid, step size, and phase information:

    data = {
        'meta':   {'xcells', 'ycells', 'xstep', 'ystep', 'phases', ...},
        'euler1': (ny, nx) float32, degrees
        'euler2': (ny, nx) float32, degrees
        'euler3': (ny, nx) float32, degrees
        'phase':  (ny, nx) uint8 phase indicator (0 = unindexed)
        'bc':     (ny, nx) band contrast
        'bs':     (ny, nx) band slope
        'mad':    (ny, nx) mean angular deviation
        'EDX_*':  (ny, nx) EDX windows if present (cpr only)
    }

Use ``read_ebsd`` as the entry point — it dispatches on file extension.
"""

from __future__ import annotations

import re as _re
from pathlib import Path
from typing import Any

import numpy as np


# --- CTF (text format) -----------------------------------------------------

def read_ctf(filepath: str | Path) -> dict[str, Any]:
    """Read an Oxford Instruments .ctf (Channel Text File) EBSD dataset."""
    import pandas as pd  # heavy; keep local

    filepath = Path(filepath)
    with open(filepath, "r", errors="replace") as f:
        lines = f.readlines()

    meta: dict[str, Any] = {"filepath": str(filepath), "name": filepath.stem}
    phases: dict[int, list[str]] = {}
    header_line: int | None = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("XCells"):
            meta["xcells"] = int(stripped.split("\t")[1])
        elif stripped.startswith("YCells"):
            meta["ycells"] = int(stripped.split("\t")[1])
        elif stripped.startswith("XStep"):
            meta["xstep"] = float(stripped.split("\t")[1])
        elif stripped.startswith("YStep"):
            meta["ystep"] = float(stripped.split("\t")[1])
        elif stripped.startswith("Phases"):
            n_phases = int(stripped.split("\t")[1])
            for j in range(1, n_phases + 1):
                phases[j] = lines[i + j].strip().split("\t")
            meta["phases"] = phases
            meta["n_phases"] = n_phases
        elif stripped.startswith("Phase\tX\tY"):
            header_line = i
            break

    if header_line is None:
        raise ValueError(f"Could not find data header in {filepath}")

    df = pd.read_csv(
        filepath, sep="\t", skiprows=header_line, engine="python",
        on_bad_lines="skip",
    )
    nx, ny = meta["xcells"], meta["ycells"]
    expected = nx * ny
    if len(df) != expected:
        if len(df) > expected:
            df = df.iloc[:expected]
        else:
            ny_adj = len(df) // nx if nx > 0 else ny
            meta["ycells"] = ny_adj
            ny = ny_adj

    data: dict[str, Any] = {"meta": meta}
    shape = (ny, nx)
    for col in df.columns:
        try:
            data[col] = df[col].values.reshape(shape)
        except ValueError:
            data[col] = df[col].values

    data["euler1"] = data.get("Euler1")
    data["euler2"] = data.get("Euler2")
    data["euler3"] = data.get("Euler3")
    data["bc"]     = data.get("BC")
    data["bs"]     = data.get("BS")
    data["mad"]    = data.get("MAD")
    data["phase"]  = data.get("Phase")
    return data


# --- CPR/CRC (binary format) ----------------------------------------------

# Field-code mapping from Oxford Instruments Channel 5 format. Taken from the
# defdap OxfordBinaryLoader source (no hard phase-construction step, so it
# works for any Laue group).
_CPR_FIELD_LOOKUP: dict[int, tuple[str, str]] = {
    3:  ("ph1",      "float32"),   # Euler1 phi1 (radians)
    4:  ("phi",      "float32"),   # Euler2 Phi  (radians)
    5:  ("ph2",      "float32"),   # Euler3 phi2 (radians)
    6:  ("MAD",      "float32"),   # mean angular deviation
    7:  ("BC",       "uint8"),     # band contrast
    8:  ("BS",       "uint8"),     # band slope
    10: ("numBands", "uint8"),
    11: ("AFI",      "uint8"),
    12: ("IB6",      "float32"),
}


def _parse_cpr_ini(cpr_path: Path) -> dict[str, dict[str, str]]:
    """Parse the ASCII INI-style .cpr header into nested dicts."""
    text = cpr_path.read_text(errors="replace")
    groups: dict[str, dict[str, str]] = {}
    current: str | None = None
    group_re = _re.compile(r"^\[(.+)\]\s*$")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        m = group_re.match(line)
        if m:
            current = m.group(1)
            groups.setdefault(current, {})
            continue
        if current is None:
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            groups[current][k.strip()] = v.strip()
    return groups


def read_cpr(filepath: str | Path) -> dict[str, Any]:
    """Read an Oxford Instruments .cpr/.crc file pair."""
    filepath = Path(filepath)
    cpr_path = filepath.with_suffix(".cpr")
    crc_path = filepath.with_suffix(".crc")
    if not cpr_path.exists():
        raise FileNotFoundError(f"CPR header not found: {cpr_path}")
    if not crc_path.exists():
        raise FileNotFoundError(f"CRC binary not found: {crc_path}")

    groups = _parse_cpr_ini(cpr_path)
    job = groups.get("Job", {})
    nx = int(job["xCells"])
    ny = int(job["yCells"])
    xstep = float(job.get("GridDistX", 1.0))
    ystep = float(job.get("GridDistY", xstep))

    # Phase metadata, shaped like ctf for downstream compatibility.
    pg = groups.get("Phases", {})
    n_phases = int(pg.get("Count", 0))
    phases: dict[int, list[str]] = {}
    for i in range(1, n_phases + 1):
        pi = groups.get(f"Phase{i}", {})
        lattice = f"{pi.get('a', '?')};{pi.get('b', '?')};{pi.get('c', '?')}"
        angles  = f"{pi.get('alpha', '?')};{pi.get('beta', '?')};{pi.get('gamma', '?')}"
        phases[i] = [lattice, angles,
                     pi.get("StructureName", f"phase_{i}"),
                     pi.get("LaueGroup", "?"), pi.get("SpaceGroup", "?")]

    edx_group = groups.get("EDX Windows", {})
    edx_names: dict[int, str] = {}
    if edx_group:
        n_edx = int(edx_group.get("Count", 0))
        for i in range(1, n_edx + 1):
            edx_names[100 + i] = edx_group.get(f"Window{i}", f"edx{i}")

    fg = groups.get("Fields", {})
    n_fields = int(fg.get("Count", 0))
    dtype_items: list[tuple[str, str]] = [("phase", "uint8")]
    unknown = 0
    for i in range(1, n_fields + 1):
        code = int(fg[f"Field{i}"])
        if code in _CPR_FIELD_LOOKUP:
            dtype_items.append(_CPR_FIELD_LOOKUP[code])
        elif code in edx_names:
            dtype_items.append((f"EDX_{edx_names[code]}", "float32"))
        else:
            unknown += 1
            dtype_items.append((f"unknown_{unknown}_code{code}", "float32"))

    record_dtype = np.dtype(dtype_items)
    raw = np.fromfile(str(crc_path), dtype=record_dtype, count=nx * ny)
    if raw.size != nx * ny:
        raw = raw[:nx * ny]

    shape = (ny, nx)
    data: dict[str, Any] = {
        "meta": {
            "filepath": str(filepath),
            "name": filepath.stem,
            "xcells": nx, "ycells": ny,
            "xstep": xstep, "ystep": ystep,
            "phases": phases, "n_phases": n_phases,
            "format": "cpr",
        },
        "phase": raw["phase"].reshape(shape),
    }

    # Euler in radians -> degrees (matching the ctf convention).
    for raw_key, deg_key in [("ph1", "euler1"), ("phi", "euler2"), ("ph2", "euler3")]:
        if raw_key in raw.dtype.names:
            data[deg_key] = np.rad2deg(raw[raw_key].reshape(shape)).astype(np.float32)

    for src, dst in [("MAD", "mad"), ("BC", "bc"), ("BS", "bs"),
                      ("numBands", "bands")]:
        if src in raw.dtype.names:
            data[dst] = raw[src].reshape(shape)
    for name in raw.dtype.names:
        if name.startswith("EDX_"):
            data[name] = raw[name].reshape(shape)

    return data


def read_ebsd(filepath: str | Path) -> dict[str, Any]:
    """Dispatch on file extension. Accepts ``.ctf``, ``.cpr``, or ``.crc``."""
    ext = Path(filepath).suffix.lower()
    if ext == ".ctf":
        return read_ctf(filepath)
    if ext in (".cpr", ".crc"):
        return read_cpr(filepath)
    raise ValueError(f"Unsupported EBSD extension: {ext}")


def split_stitched(
    ebsd: dict,
    *,
    col_frac_thr: float = 0.4,
    row_frac_thr: float = 0.4,
    buffer_px: int = 3,
    min_side_px: int = 64,
) -> list[dict]:
    """Split an EBSD map with interior stitch columns into independent tiles.

    Stitch boundaries in Channel 5 / AZtec EBSD acquisitions show up as a
    thin column (or several) where almost every pixel is unindexed
    (``phase == 0``). This function detects those columns, plus the
    typical low-indexing strips at the top/bottom of a map, and returns
    a list of cropped sub-EBSD dicts — one per clean tile.

    The returned dicts have the same keys as ``ebsd`` (2-D arrays are
    cropped, 1-D/non-spatial values pass through) and a ``meta`` whose
    ``name`` is suffixed with ``_tileN`` and whose ``xcells``/``ycells``
    reflect the new tile dimensions. ``meta['crop']`` records
    ``{'rows': (r0, r1), 'cols': (c0, c1)}`` relative to the input.

    Parameters
    ----------
    ebsd : dict from :func:`read_ebsd`.
    col_frac_thr : absolute unindexed-fraction threshold for calling a
        column a stitch. Also requires > 3× the median column fraction
        to stay robust against high-background samples.
    row_frac_thr : same, for trimming top/bottom rows.
    buffer_px : pixels to drop on each side of a stitch column (so the
        tile boundary isn't right at the noisy stitch edge).
    min_side_px : tiles smaller than this on any side are discarded.

    Returns
    -------
    list of dict. Empty if the entire map is bad; one element (= the
    input, possibly row-trimmed) if no interior stitch is found.
    """
    pm = ebsd["phase"]
    ny, nx = pm.shape

    col_unidx = (pm == 0).mean(axis=0)
    row_unidx = (pm == 0).mean(axis=1)
    median_c = float(np.median(col_unidx))
    median_r = float(np.median(row_unidx))
    bad_col = col_unidx > max(col_frac_thr, 3.0 * median_c)
    bad_row = row_unidx > max(row_frac_thr, 3.0 * median_r)

    # Identify runs of bad columns that sit *inside* the map (not at the
    # edges — those are just border effects, not stitches).
    cuts: list[tuple[int, int]] = []
    i = 0
    while i < nx:
        if bad_col[i]:
            j = i
            while j < nx and bad_col[j]:
                j += 1
            if i > buffer_px and j < nx - buffer_px:
                cuts.append((i, j))
            i = j
        else:
            i += 1

    # Build column ranges — one per tile, skipping the stitch + buffer.
    if cuts:
        col_ranges, prev = [], 0
        for (a, b) in cuts:
            col_ranges.append((prev, a))
            prev = b
        col_ranges.append((prev, nx))
    else:
        col_ranges = [(0, nx)]

    # Trim top/bottom rows that are mostly unindexed.
    top = 0
    while top < ny and bad_row[top]:
        top += 1
    bot = ny
    while bot > 0 and bad_row[bot - 1]:
        bot -= 1
    # If trimming would remove more than half the rows, back off — the
    # whole map is probably noisy and we don't want to over-crop.
    if bot - top < 0.5 * ny:
        top, bot = 0, ny

    tiles: list[dict] = []
    for ti, (c0, c1) in enumerate(col_ranges):
        # Pull in from the stitch side by the buffer.
        if ti > 0:
            c0 += buffer_px
        if ti < len(col_ranges) - 1:
            c1 -= buffer_px
        if (c1 - c0) < min_side_px or (bot - top) < min_side_px:
            continue

        sub: dict = {}
        for k, v in ebsd.items():
            if k == "meta":
                continue
            if hasattr(v, "shape") and v.ndim == 2:
                sub[k] = v[top:bot, c0:c1].copy()
            else:
                sub[k] = v

        new_meta = dict(ebsd["meta"])
        new_meta["xcells"] = c1 - c0
        new_meta["ycells"] = bot - top
        orig_name = ebsd["meta"].get("name", "ebsd")
        new_meta["name"] = f"{orig_name}_tile{ti + 1}"
        new_meta["crop"] = {"rows": (top, bot), "cols": (c0, c1)}
        sub["meta"] = new_meta
        tiles.append(sub)

    return tiles


# --- Simple scalar-field extractors ---------------------------------------

def phase_mask(ebsd: dict, phase_id: int) -> np.ndarray:
    """Boolean mask where the phase index matches ``phase_id``."""
    return ebsd["phase"] == phase_id


def find_phase_id(ebsd: dict, name_substrings: list[str]) -> int | None:
    """First phase index whose StructureName contains any of the given
    substrings (case-insensitive). Returns None if no match."""
    phases = ebsd["meta"].get("phases", {})
    for pid, info in phases.items():
        pname = info[2] if len(info) > 2 else ""
        if any(s.lower() in pname.lower() for s in name_substrings):
            return int(pid)
    return None


def kam(
    ebsd: dict,
    phase_id: int,
    *,
    kernel: int = 1,
    symmetry: str = "trigonal",
    grain_id: np.ndarray | None = None,
) -> np.ndarray:
    """Kernel-average misorientation (degrees) for a single phase.

    Uses orix for the symmetry-aware misorientation. For quartz pass
    ``symmetry='trigonal'`` (matches Laue group '3'); for olivine use
    ``'orthorhombic'``.

    If ``grain_id`` is supplied (a 2D array of the same shape as the ebsd
    map), neighbour pairs that cross a grain boundary are excluded from the
    average. This mirrors MTEX's default behaviour and produces an
    intra-grain KAM field that reflects dislocation substructure rather
    than inter-grain misorientation.

    Pixels not in the phase are returned as NaN; border pixels are also NaN.
    """
    try:
        from orix.quaternion import Orientation
        from orix.quaternion.symmetry import C1, C2, D2, D3, D6, C4, Oh
    except ImportError as e:
        raise ImportError(
            "kam() requires orix. Install with: pip install orix"
        ) from e

    sym_map = {
        "triclinic":    C1,
        "monoclinic":   C2,
        "orthorhombic": D2,
        "trigonal":     D3,
        "hexagonal":    D6,
        "tetragonal":   C4,
        "cubic":        Oh,
    }
    sym = sym_map.get(symmetry)
    if sym is None:
        raise ValueError(f"Unknown symmetry: {symmetry}")

    eul = np.stack([ebsd["euler1"], ebsd["euler2"], ebsd["euler3"]], axis=-1)
    mask = phase_mask(ebsd, phase_id)
    ny, nx = mask.shape
    euler_rad = np.deg2rad(eul.astype(np.float64))

    # Vectorised KAM: for each (di, dj) shift in the kernel, compute the
    # whole-grid misorientation angle in one orix call, then accumulate.
    # This is ~100x faster than the per-pixel double loop and avoids the
    # symmetry-reduction scalar-extraction quirk of Orientation.angle.
    ori = Orientation.from_euler(euler_rad.reshape(-1, 3), symmetry=sym)
    ori = ori.reshape(ny, nx)

    total = np.zeros((ny, nx), dtype=np.float64)
    count = np.zeros((ny, nx), dtype=np.int32)
    in_phase = mask.astype(bool)

    shifts = [(di, dj) for di in range(-kernel, kernel + 1)
              for dj in range(-kernel, kernel + 1) if (di, dj) != (0, 0)]

    for di, dj in shifts:
        # Overlap slices so the shifted comparison stays within bounds.
        i0, i1 = max(0, -di), min(ny, ny - di)
        j0, j1 = max(0, -dj), min(nx, nx - dj)
        if i0 >= i1 or j0 >= j1:
            continue

        a = ori[i0:i1, j0:j1]
        b = ori[i0 + di:i1 + di, j0 + dj:j1 + dj]
        # Misorientation: a * b.inv (inv = ~). Orientation supports slicing.
        mis = a * (~b)
        # mis.angle is an ndarray with the same shape as the slice.
        ang_deg = np.rad2deg(np.asarray(mis.angle))

        # Valid where both endpoints are in-phase.
        ok = in_phase[i0:i1, j0:j1] & in_phase[i0 + di:i1 + di,
                                                j0 + dj:j1 + dj]
        # MTEX-style grain-aware masking: reject pairs across grain boundary.
        if grain_id is not None:
            same_grain = (grain_id[i0:i1, j0:j1]
                          == grain_id[i0 + di:i1 + di, j0 + dj:j1 + dj])
            ok &= same_grain
        total[i0:i1, j0:j1] = np.where(ok, total[i0:i1, j0:j1] + ang_deg,
                                            total[i0:i1, j0:j1])
        count[i0:i1, j0:j1] = np.where(ok, count[i0:i1, j0:j1] + 1,
                                            count[i0:i1, j0:j1])

    out = np.full((ny, nx), np.nan, dtype=np.float32)
    valid = (count > 0) & in_phase
    out[valid] = (total[valid] / count[valid]).astype(np.float32)
    return out


# --- Log-orientation, Nye tensor, SVD anisotropy, Helmholtz --------------

_SYMMETRY_GROUPS = {
    "triclinic", "monoclinic", "orthorhombic", "trigonal",
    "hexagonal", "tetragonal", "cubic",
}

# Rough mapping from common rock-forming mineral names to orix symmetry
# keywords. Intentionally permissive — the keys are substrings; first match
# wins. Unknown phases fall back to "triclinic" (no symmetry reduction).
_MINERAL_SYMMETRY: tuple[tuple[str, str], ...] = (
    ("quartz",         "trigonal"),     # α-quartz, trigonal D3
    ("orthoclase",     "monoclinic"),   # K-feldspar, C2/m
    ("microcline",     "triclinic"),
    ("albite",         "triclinic"),
    ("anorthite",      "triclinic"),
    ("plagioclase",    "triclinic"),
    ("sanidine",       "monoclinic"),
    ("biotite",        "monoclinic"),
    ("muscovite",      "monoclinic"),
    ("phlogopite",     "monoclinic"),
    ("chlorite",       "monoclinic"),
    ("hornblende",     "monoclinic"),
    ("amphibole",      "monoclinic"),
    ("tremolite",      "monoclinic"),
    ("diopside",       "monoclinic"),
    ("augite",         "monoclinic"),
    ("olivine",        "orthorhombic"),
    ("forsterite",     "orthorhombic"),
    ("fayalite",       "orthorhombic"),
    ("enstatite",      "orthorhombic"),
    ("clinopyroxene",  "monoclinic"),
    ("orthopyroxene",  "orthorhombic"),
    ("calcite",        "trigonal"),
    ("dolomite",       "trigonal"),
    ("pyrite",         "cubic"),
    ("garnet",         "cubic"),
    ("magnetite",      "cubic"),
    ("ilmenite",       "trigonal"),
    ("rutile",         "tetragonal"),
    ("zircon",         "tetragonal"),
    ("apatite",        "hexagonal"),
)


def guess_symmetry(phase_name: str) -> str:
    """Return an orix-compatible symmetry keyword for a mineral phase name.

    Matches substrings case-insensitively against a short catalogue of
    rock-forming minerals. Falls back to 'triclinic' (= C1, no symmetry
    reduction) if no match. Safe to call for unknown phases — the fallback
    just means log-orientation vectors are raw axis-angle from the
    reference orientation, with no symmetry-equivalent minimisation.
    """
    q = phase_name.lower()
    for key, sym in _MINERAL_SYMMETRY:
        if key in q:
            return sym
    return "triclinic"


def _orix_symmetry(name: str):
    from orix.quaternion.symmetry import C1, C2, D2, D3, D6, C4, Oh
    return {
        "triclinic":    C1,
        "monoclinic":   C2,
        "orthorhombic": D2,
        "trigonal":     D3,
        "hexagonal":    D6,
        "tetragonal":   C4,
        "cubic":        Oh,
    }[name]


def log_orientation(
    ebsd: dict,
    phase_id: int,
    *,
    symmetry: str = "trigonal",
    reference: str | None = "median",
) -> np.ndarray:
    """Log-orientation (axis-angle) vector field relative to a reference.

    For each pixel of the chosen phase, compute the symmetry-reduced
    misorientation from a reference orientation and return it as a 3-vector
    ``v = (θ/2) · n̂`` (axis-angle representation; units radians).

    Parameters
    ----------
    ebsd : dict
        Dict returned by :func:`read_ebsd`.
    phase_id : int
        Phase index to extract.
    symmetry : str
        One of ``triclinic, monoclinic, orthorhombic, trigonal,
        hexagonal, tetragonal, cubic``. Quartz = ``'trigonal'``.
    reference : {"median", "mean", None}
        Reference orientation for log computation. ``median`` picks the
        pixel with the minimum total misorientation to all others (cheap
        approximation: use mean-quaternion centroid). ``None`` uses the
        identity orientation (raw log of the absolute orientation).

    Returns
    -------
    logq : ndarray, shape (ny, nx, 3)
        Axis-angle vectors. Pixels not in the requested phase get NaN.
    """
    from orix.quaternion import Orientation

    if symmetry not in _SYMMETRY_GROUPS:
        raise ValueError(f"unknown symmetry: {symmetry}")
    sym = _orix_symmetry(symmetry)

    eul = np.stack([ebsd["euler1"], ebsd["euler2"], ebsd["euler3"]], axis=-1)
    ny, nx = eul.shape[:2]
    mask = phase_mask(ebsd, phase_id)
    euler_rad = np.deg2rad(eul.astype(np.float64))

    ori = Orientation.from_euler(euler_rad.reshape(-1, 3), symmetry=sym)
    ori = ori.reshape(ny, nx)

    if reference is None:
        ref = Orientation.from_euler(np.zeros((1, 3)), symmetry=sym)
        ref_scalar = ref[0]
    else:
        # Cheap "central" reference: pick the pixel whose misorientation to
        # all other in-phase pixels has the smallest mean angle. For large
        # fields that's O(N^2); as an approximation we subsample the phase.
        in_phase_idx = np.argwhere(mask)
        if len(in_phase_idx) == 0:
            raise ValueError(f"no pixels in phase {phase_id}")
        n_sub = min(200, len(in_phase_idx))
        rng = np.random.default_rng(0)
        sub = in_phase_idx[rng.choice(len(in_phase_idx), n_sub, replace=False)]
        sub_ori = Orientation.from_euler(
            euler_rad[sub[:, 0], sub[:, 1]], symmetry=sym)
        # Pairwise mean angle: for each candidate i, mean over j
        best_i, best_mean = 0, np.inf
        for i in range(n_sub):
            mis = sub_ori[i] * (~sub_ori)
            ang = np.asarray(mis.angle)
            m = float(np.mean(ang))
            if m < best_mean:
                best_mean = m; best_i = i
        ref_r, ref_c = sub[best_i]
        ref_scalar = ori[ref_r, ref_c]

    # Misorientation from reference: ref^-1 * q. Flatten for speed.
    mis_flat = (~ref_scalar) * ori.reshape(-1)
    # axis-angle: angle in rad, then axis = (x, y, z)/sin(angle/2)
    ang = np.asarray(mis_flat.angle)          # (N,)
    # orix uses quaternion (a, b, c, d) with a = cos(angle/2)
    q = np.asarray(mis_flat.data)             # (N, 4)
    s = np.sin(ang / 2.0)
    s_safe = np.where(s > 1e-12, s, 1.0)
    axis = q[:, 1:] / s_safe[:, None]        # (N, 3)
    logq_flat = (ang[:, None] / 2.0) * axis  # (N, 3)
    logq = logq_flat.reshape(ny, nx, 3).astype(np.float32)

    # Mask out-of-phase pixels
    logq[~mask] = np.nan
    return logq


def nye_tensor(
    logq: np.ndarray,
    step: float = 1.0,
    grain_id: np.ndarray | None = None,
) -> np.ndarray:
    """Nye-tensor proxy from a log-orientation field.

    For a 2D map with only in-plane derivatives, we compute the curl-like
    combinations of the log-orientation gradient. The returned (ny, nx, 3, 3)
    tensor has non-zero entries only in the j=3 (out-of-plane) slab:

        α_i1 ≡ ∂(log q)_i / ∂y
        α_i2 ≡ -∂(log q)_i / ∂x
        α_i3 = 0   (no z derivative in a 2D map)

    and the "curl-of-curl" scalar Frobenius norm ``||α||_F`` is the standard
    GND-density proxy (see :func:`gnd_density`).

    NaN-safe: non-phase pixels propagate NaN in gradients.

    If ``grain_id`` is provided, pixels whose central-difference stencil
    spans two different grains are NaN'd — otherwise the "gradient" across
    a grain boundary is a discontinuity, not a physical derivative, and
    produces a spurious GND spike. This matches MTEX's behaviour.

    Parameters
    ----------
    logq : ndarray, shape (ny, nx, 3)
    step : float
        Spatial step size (same units as you want the gradient in — usually
        µm for EBSD).
    grain_id : ndarray, optional
        Same shape (ny, nx). Connected-component grain labels; 0 = unindexed.
    """
    ny, nx, nc = logq.shape
    if nc != 3:
        raise ValueError("expected 3 channels for log-orientation")
    alpha = np.zeros((ny, nx, 3, 3), dtype=np.float32)
    for i in range(3):
        gy, gx = np.gradient(logq[..., i], step, step)
        alpha[..., i, 0] = gy       # α_i1
        alpha[..., i, 1] = -gx      # α_i2

    if grain_id is not None:
        if grain_id.shape != (ny, nx):
            raise ValueError("grain_id shape must match logq[:,:,:3] spatial shape")
        same_up    = (grain_id == np.roll(grain_id, 1, axis=0))
        same_down  = (grain_id == np.roll(grain_id, -1, axis=0))
        same_left  = (grain_id == np.roll(grain_id, 1, axis=1))
        same_right = (grain_id == np.roll(grain_id, -1, axis=1))
        # Central-difference stencil touches all 4 neighbours; any cross-grain
        # contact invalidates the gradient at this pixel.
        stencil_ok = same_up & same_down & same_left & same_right
        # np.roll wraps the borders; zero them out explicitly.
        stencil_ok[0] = False; stencil_ok[-1] = False
        stencil_ok[:, 0] = False; stencil_ok[:, -1] = False
        alpha[~stencil_ok] = np.nan
    return alpha


def gnd_density(
    alpha: np.ndarray,
    *,
    burgers: float = 1.0,
) -> np.ndarray:
    """Scalar GND-density proxy ``||α||_F / b`` (dimensions: 1/length).

    If your step is in µm and you want 1/m², divide `burgers` accordingly.
    """
    return np.sqrt(np.sum(alpha * alpha, axis=(-2, -1))) / burgers


def svd_anisotropy(
    logq: np.ndarray,
    *,
    patch_radius: int = 2,
) -> dict[str, np.ndarray]:
    """Local-patch SVD of the log-orientation field.

    For each pixel, collect its ``(2r+1)^2`` neighbours into a
    ``(n_pix, 3)`` matrix, run SVD, and report:

    - ``sigma[0,1,2]``: the three singular values (ny, nx) each
    - ``c_l``, ``c_p``, ``c_s``: Westin linear / planar / spherical coefficients
    - ``frob``: Frobenius norm of the patch matrix

    Pixels whose patch touches the image border or contains any NaN are NaN.
    Returns a dict of (ny, nx) float32 arrays.
    """
    ny, nx, nc = logq.shape
    if nc != 3:
        raise ValueError("expected 3 channels for log-orientation")
    r = patch_radius
    pw = 2 * r + 1

    sigma = np.full((3, ny, nx), np.nan, dtype=np.float32)
    frob  = np.full((ny, nx), np.nan, dtype=np.float32)

    # Simple double loop; patches are small so this stays fast at 125x125.
    for i in range(r, ny - r):
        for j in range(r, nx - r):
            patch = logq[i - r:i + r + 1, j - r:j + r + 1, :]
            if not np.isfinite(patch).all():
                continue
            M = patch.reshape(-1, 3)
            s = np.linalg.svd(M, compute_uv=False)   # (3,)
            sigma[0, i, j] = s[0]
            sigma[1, i, j] = s[1]
            sigma[2, i, j] = s[2]
            frob[i, j] = float(np.linalg.norm(M, ord="fro"))

    s1, s2, s3 = sigma[0], sigma[1], sigma[2]
    denom = np.where(s1 > 0, s1, 1.0)
    c_l = (s1 - s2) / denom
    c_p = (s2 - s3) / denom
    c_s = s3 / denom
    return {
        "sigma1": s1, "sigma2": s2, "sigma3": s3,
        "c_l": c_l.astype(np.float32),
        "c_p": c_p.astype(np.float32),
        "c_s": c_s.astype(np.float32),
        "frob": frob,
    }


# --- Grain reconstruction + derived scalar fields -------------------------
#
# A "grain" is a connected region of same-phase pixels whose pairwise
# misorientations are all below a threshold (default 10°). This follows the
# MTEX convention.  We reduce it to connected-components on a sparse graph
# where every same-phase, low-misorientation neighbour pair is an edge.

def grain_reconstruction(
    ebsd: dict,
    *,
    threshold_deg: float = 10.0,
    connectivity: int = 8,
    symmetry_by_phase: dict[int, str] | None = None,
) -> np.ndarray:
    """Per-pixel grain ID via connected components on a misorientation graph.

    Pixels belong to the same grain iff they share a phase AND their
    symmetry-reduced misorientation is below ``threshold_deg``. Unindexed
    pixels (phase == 0) get grain ID 0. Grain IDs start at 1.

    Parameters
    ----------
    ebsd : dict
        EBSD dict from :func:`read_ebsd`.
    threshold_deg : float
        HAGB / LAGB cutoff in degrees. 10° is the MTEX default; lower for
        subgrain analysis.
    connectivity : int
        4 (orthogonal neighbours only) or 8 (orthogonal + diagonal).
    symmetry_by_phase : dict | None
        Map phase_id -> orix symmetry keyword. If None, each phase's
        symmetry is guessed from its name via :func:`guess_symmetry`.

    Returns
    -------
    grain_id : (ny, nx) int32 array, 0 where unindexed.
    """
    try:
        from orix.quaternion import Orientation
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
    except ImportError as e:
        raise ImportError(
            "grain_reconstruction requires orix and scipy"
        ) from e

    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    phase = ebsd["phase"]
    ny, nx = phase.shape
    N = ny * nx
    threshold_rad = np.deg2rad(threshold_deg)

    # Pre-compute orientations once per phase so we can index them cheaply.
    # For each phase, we need one flat list of Orientation objects in raster
    # order (len = ny*nx). Pixels not in that phase get identity + flag.
    phases = ebsd["meta"].get("phases", {})
    if symmetry_by_phase is None:
        symmetry_by_phase = {
            int(pid): guess_symmetry(info[2])
            for pid, info in phases.items()
        }

    euler_rad = np.deg2rad(np.stack(
        [ebsd["euler1"], ebsd["euler2"], ebsd["euler3"]], axis=-1
    ).astype(np.float64))

    # Build a list of edges (u, v) where u < v and both pixels belong to the
    # same phase and their symmetry-reduced misorientation angle is below
    # the threshold.
    edge_rows: list[np.ndarray] = []
    edge_cols: list[np.ndarray] = []

    shifts = [(0, 1), (1, 0)]
    if connectivity == 8:
        shifts += [(1, 1), (1, -1)]

    for pid, info in phases.items():
        pid_int = int(pid)
        sym_name = symmetry_by_phase.get(pid_int, "triclinic")
        sym = _orix_symmetry(sym_name)

        in_phase = (phase == pid_int)
        if in_phase.sum() < 2:
            continue

        # Build Orientation objects for all pixels (cheap) so slicing stays
        # valid per-phase.
        ori = Orientation.from_euler(euler_rad.reshape(-1, 3), symmetry=sym)
        ori = ori.reshape(ny, nx)

        for di, dj in shifts:
            i0, i1_ = max(0, -di), min(ny, ny - di)
            j0, j1_ = max(0, -dj), min(nx, nx - dj)
            if i0 >= i1_ or j0 >= j1_:
                continue

            a_mask = in_phase[i0:i1_, j0:j1_]
            b_mask = in_phase[i0 + di:i1_ + di, j0 + dj:j1_ + dj]
            both = a_mask & b_mask
            if not np.any(both):
                continue

            a = ori[i0:i1_, j0:j1_]
            b = ori[i0 + di:i1_ + di, j0 + dj:j1_ + dj]
            mis = a * (~b)
            ang = np.asarray(mis.angle)

            keep = both & (ang < threshold_rad)
            if not np.any(keep):
                continue

            ii, jj = np.nonzero(keep)
            # Map local (ii, jj) back to absolute pixel indices in the
            # raster-order adjacency graph.
            u_row = i0 + ii;         u_col = j0 + jj
            v_row = u_row + di;      v_col = u_col + dj
            edge_rows.append((u_row * nx + u_col).astype(np.int64))
            edge_cols.append((v_row * nx + v_col).astype(np.int64))

    if not edge_rows:
        # No edges => every indexed pixel is its own grain.
        grain_id = np.zeros((ny, nx), dtype=np.int32)
        indexed = phase > 0
        grain_id[indexed] = np.arange(1, indexed.sum() + 1)
        return grain_id

    rows = np.concatenate(edge_rows)
    cols = np.concatenate(edge_cols)
    # Symmetric graph: add reverse edges so connected_components sees both.
    adj = coo_matrix(
        (np.ones(len(rows) * 2, dtype=np.int8),
         (np.concatenate([rows, cols]), np.concatenate([cols, rows]))),
        shape=(N, N),
    ).tocsr()
    n_comp, labels = connected_components(adj, directed=False)

    # Collapse singleton (unindexed) pixels to grain 0 and compact IDs so
    # they start at 1 for indexed pixels.
    indexed = (phase.ravel() > 0)
    grain_id_flat = np.zeros(N, dtype=np.int32)
    if indexed.any():
        lbl_indexed = labels[indexed]
        # Compact label space
        _, compact = np.unique(lbl_indexed, return_inverse=True)
        grain_id_flat[indexed] = compact + 1
    return grain_id_flat.reshape(ny, nx)


def gos(
    ebsd: dict,
    grain_id: np.ndarray,
    *,
    symmetry_by_phase: dict[int, str] | None = None,
    min_grain_size: int = 5,
) -> np.ndarray:
    """Grain orientation spread (degrees) per pixel.

    For each grain, compute its mean orientation as the L² quaternion mean,
    then for each pixel return the angle between the pixel's orientation
    and the grain mean, symmetry-reduced per phase.

    Grains smaller than ``min_grain_size`` and unindexed pixels get NaN.
    """
    from orix.quaternion import Orientation

    phase = ebsd["phase"]
    ny, nx = phase.shape
    out = np.full((ny, nx), np.nan, dtype=np.float32)

    phases = ebsd["meta"].get("phases", {})
    if symmetry_by_phase is None:
        symmetry_by_phase = {
            int(pid): guess_symmetry(info[2])
            for pid, info in phases.items()
        }

    euler_rad = np.deg2rad(np.stack(
        [ebsd["euler1"], ebsd["euler2"], ebsd["euler3"]], axis=-1
    ).astype(np.float64))

    unique_gids = np.unique(grain_id)
    for gid in unique_gids:
        if gid == 0:
            continue
        mask = (grain_id == gid)
        n = int(mask.sum())
        if n < min_grain_size:
            continue
        pix_phase = int(phase[mask][0])
        sym = _orix_symmetry(symmetry_by_phase.get(pix_phase, "triclinic"))

        # Orientations for this grain
        eu = euler_rad[mask]
        ori = Orientation.from_euler(eu, symmetry=sym)
        # Quaternion mean: normalise the component-wise mean quaternion.
        q = np.asarray(ori.data)          # (n, 4)
        q_mean = q.mean(axis=0)
        q_mean /= np.linalg.norm(q_mean) + 1e-15
        ori_mean = Orientation(q_mean.reshape(1, 4), symmetry=sym)[0]
        mis = ori * (~ori_mean)
        ang_deg = np.rad2deg(np.asarray(mis.angle))
        # Write back into the 2D grid
        out[mask] = ang_deg.astype(np.float32)

    return out


def c_axis_deviation(
    ebsd: dict,
    phase_id: int,
    *,
    symmetry: str = "trigonal",
    reference_direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    dominant: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """Angular deviation of each pixel's c-axis from the phase-dominant direction.

    For each indexed pixel of the chosen phase, compute the crystal
    ``reference_direction`` (default [001] = c-axis) rotated into the sample
    frame by the pixel's orientation. Then find the phase-dominant c-axis
    (mean resultant vector on S²) and return the per-pixel angular deviation
    from it in degrees.

    Parameters
    ----------
    dominant : {"mean", "max"}
        Method for finding the phase's dominant c-axis.
        ``"mean"``: resultant vector of all pixel c-axes (handles axial
        ambiguity by reflecting every axis to the +z hemisphere first).
        ``"max"``: kernel density estimate on S² and take its peak (slower).

    Returns
    -------
    dev_deg : (ny, nx) float32 — NaN outside phase
    dominant_axis : (3,) the sample-frame c-axis used as reference
    """
    # scipy's Rotation handles Euler-to-rotation conversions and vector
    # application cleanly. Bunge ZXZ intrinsic convention matches the
    # channel 5 / MTEX EBSD convention we read from .cpr / .ctf files.
    from scipy.spatial.transform import Rotation as ScipyRot

    eul = np.stack([ebsd["euler1"], ebsd["euler2"], ebsd["euler3"]], axis=-1)
    ny, nx = eul.shape[:2]
    mask = phase_mask(ebsd, phase_id)
    euler_rad = np.deg2rad(eul.astype(np.float64))

    R = ScipyRot.from_euler("ZXZ", euler_rad.reshape(-1, 3))
    ref = np.array(reference_direction, dtype=np.float64)
    ref /= np.linalg.norm(ref) + 1e-15
    # Crystal reference direction expressed in sample frame per pixel.
    axes_arr = R.apply(ref).reshape(ny, nx, 3)

    # Keep only in-phase pixels for the dominant-axis estimate.
    A = axes_arr[mask]
    if A.size == 0:
        raise ValueError(f"no pixels in phase {phase_id}")

    # Resolve axial ambiguity: flip every axis to the +z hemisphere.
    A = A * np.where(A[:, 2:3] < 0, -1.0, 1.0)

    if dominant == "mean":
        v = A.mean(axis=0)
        v /= np.linalg.norm(v) + 1e-15
    elif dominant == "max":
        # Simple spherical KDE: grid the hemisphere, find the maximum.
        n_grid = 64
        lat = np.linspace(0, np.pi / 2, n_grid)
        lon = np.linspace(0, 2 * np.pi, n_grid)
        LAT, LON = np.meshgrid(lat, lon, indexing="ij")
        gx = np.sin(LAT) * np.cos(LON)
        gy = np.sin(LAT) * np.sin(LON)
        gz = np.cos(LAT)
        grid = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
        # kernel: cos(angle) > threshold. Pick bandwidth 5°.
        bw = np.cos(np.deg2rad(5.0))
        density = (A @ grid.T > bw).sum(axis=0)
        v = grid[density.argmax()]
    else:
        raise ValueError("dominant must be 'mean' or 'max'")

    # Per-pixel angular deviation from v, in degrees.
    dots = (axes_arr @ v)
    # Axial symmetry: use |dot|
    dots = np.clip(np.abs(dots), 0.0, 1.0)
    dev = np.rad2deg(np.arccos(dots)).astype(np.float32)
    out = np.full((ny, nx), np.nan, dtype=np.float32)
    out[mask] = dev[mask]
    return out, v


def phase_agnostic_kam(
    ebsd: dict,
    *,
    kernel: int = 1,
    grain_id: np.ndarray | None = None,
) -> np.ndarray:
    """Kernel Average Misorientation with no symmetry reduction (C1).

    Every indexed pixel (phase > 0) is treated as one phase with triclinic
    symmetry, so the per-pixel value is the raw angular distance in
    orientation space from its neighbours. Useful when a single misorientation
    field that spans multiple minerals is wanted — e.g. as the scalar input
    to a recurrence network on the whole sample.

    Pass ``grain_id`` to exclude across-boundary neighbour pairs
    (MTEX-consistent intra-grain KAM).
    """
    fake = dict(ebsd)
    fake["phase"] = (ebsd["phase"] > 0).astype(np.uint8)
    fake["meta"] = dict(ebsd["meta"])
    fake["meta"]["phases"] = {1: ["?", "?", "all_indexed", "?", "?"]}
    return kam(fake, phase_id=1, kernel=kernel, symmetry="triclinic",
                grain_id=grain_id)


# --- Multifractal structure-function analysis -----------------------------

def structure_function(
    field: np.ndarray,
    q_values: np.ndarray,
    scales: np.ndarray,
) -> dict[str, np.ndarray]:
    """Structure functions S_q(ℓ) = ⟨|X(x+ℓ) − X(x)|^q⟩ of a 2D field.

    Averages over finite-valued increments in the +x and +y directions.
    NaN pixels in the input are skipped at each scale.

    Returns
    -------
    dict with keys:
      'Sq' : (n_q, n_scales) — structure function values
      'scales', 'q_values' — echoed back
      'zeta' : (n_q,) — scaling exponents, from log-log slope of S_q(ℓ)
    """
    q_values = np.asarray(q_values, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.int64)
    Sq = np.full((len(q_values), len(scales)), np.nan, dtype=np.float64)

    for si, ell in enumerate(scales):
        diffs = []
        if ell < field.shape[0]:
            d = field[ell:, :] - field[:-ell, :]
            diffs.append(d[np.isfinite(d)])
        if ell < field.shape[1]:
            d = field[:, ell:] - field[:, :-ell]
            diffs.append(d[np.isfinite(d)])
        if not diffs:
            continue
        vals = np.concatenate(diffs)
        if vals.size < 20:
            continue
        # Add a small floor so q < 0 doesn't blow up on zero increments
        # (plentiful in quantised fields like raw KAM).
        abs_vals = np.abs(vals) + 1e-12
        for qi, q in enumerate(q_values):
            if q == 0:
                Sq[qi, si] = np.exp(np.mean(np.log(abs_vals)))
            else:
                Sq[qi, si] = np.mean(abs_vals ** q)

    # ζ(q) from the log-log slope across scales
    zeta = np.full(len(q_values), np.nan, dtype=np.float64)
    log_l = np.log(scales.astype(np.float64))
    for qi in range(len(q_values)):
        y = Sq[qi]
        ok = np.isfinite(y) & (y > 0)
        if ok.sum() >= 3:
            zeta[qi] = np.polyfit(log_l[ok], np.log(y[ok]), 1)[0]

    return {
        "Sq": Sq, "scales": scales, "q_values": q_values, "zeta": zeta,
    }


def f_alpha_from_zeta(
    q_values: np.ndarray,
    zeta: np.ndarray,
    *,
    embedding_dim: int = 2,
) -> dict[str, np.ndarray]:
    """Singularity spectrum f(α) via Legendre transform of τ(q).

    Uses τ(q) = ζ(q) − q (1D convention) or τ(q) = ζ(q) − d (generalised).
    For a 2D scalar field, ``embedding_dim`` = 2 gives the correct shift.
    α = dτ/dq, f(α) = q α − τ(q).
    """
    q = np.asarray(q_values, dtype=np.float64)
    zeta = np.asarray(zeta, dtype=np.float64)
    finite = np.isfinite(zeta)
    if finite.sum() < 5:
        return {"alpha": np.array([]), "f_alpha": np.array([]),
                "tau": np.array([]), "q_values": q[finite], "zeta": zeta[finite]}
    q_f = q[finite]; z_f = zeta[finite]
    tau = z_f - embedding_dim
    # Numerical derivative α = dτ/dq
    alpha = np.gradient(tau, q_f)
    f = q_f * alpha - tau
    return {
        "alpha": alpha, "f_alpha": f, "tau": tau,
        "q_values": q_f, "zeta": z_f,
    }


def spectrum_width(alpha: np.ndarray, f_alpha: np.ndarray,
                   f_min: float = 0.0) -> float:
    """Width of the singularity spectrum at f = f_min.

    Returns α_max − α_min over points where f(α) ≥ f_min. NaN if the
    spectrum has fewer than two above-threshold points.
    """
    sel = np.isfinite(alpha) & np.isfinite(f_alpha) & (f_alpha >= f_min)
    if sel.sum() < 2:
        return float("nan")
    a = alpha[sel]
    return float(a.max() - a.min())


def helmholtz_2d(
    vx: np.ndarray,
    vy: np.ndarray,
) -> dict[str, np.ndarray]:
    """FFT-based 2D Helmholtz decomposition of a vector field.

    Returns curl-free and divergence-free parts that sum to ``(vx, vy)``.
    Assumes periodic BCs (the FFT convention); for non-periodic fields the
    decomposition has small boundary artefacts but captures the right bulk
    structure. NaNs in the input are replaced with the field mean before
    the transform.

    Returns
    -------
    dict with:
        vx_cf, vy_cf : curl-free part (gradient of a potential)
        vx_df, vy_df : divergence-free part (rotation of a stream function)
        phi          : scalar potential (such that ∇φ = curl-free part)
        psi          : stream function (such that ∇×(ψ ẑ) = div-free part)
    """
    if vx.shape != vy.shape:
        raise ValueError("vx and vy must have the same shape")
    ny, nx = vx.shape

    u = np.where(np.isfinite(vx), vx, np.nanmean(vx))
    v = np.where(np.isfinite(vy), vy, np.nanmean(vy))

    kx = 2.0 * np.pi * np.fft.fftfreq(nx)[None, :]
    ky = 2.0 * np.pi * np.fft.fftfreq(ny)[:, None]
    k2 = kx * kx + ky * ky
    k2[0, 0] = 1.0  # avoid div-zero; DC component handled below.

    U = np.fft.fft2(u)
    V = np.fft.fft2(v)

    # curl-free part = ∇φ where φ solves ∇²φ = div(u,v)
    divUV = 1j * kx * U + 1j * ky * V
    phi_hat = -divUV / k2
    phi_hat[0, 0] = 0.0
    vx_cf = np.real(np.fft.ifft2(1j * kx * phi_hat))
    vy_cf = np.real(np.fft.ifft2(1j * ky * phi_hat))

    # div-free part = (u, v) - curl-free part
    vx_df = u - vx_cf
    vy_df = v - vy_cf

    # Stream function: ∂ψ/∂y = vx_df, -∂ψ/∂x = vy_df, i.e., ∇²ψ = curl(u,v)
    curlUV = 1j * kx * V - 1j * ky * U
    psi_hat = curlUV / k2
    psi_hat[0, 0] = 0.0
    phi = np.real(np.fft.ifft2(phi_hat))
    psi = np.real(np.fft.ifft2(psi_hat))

    return {
        "vx_cf": vx_cf.astype(np.float32), "vy_cf": vy_cf.astype(np.float32),
        "vx_df": vx_df.astype(np.float32), "vy_df": vy_df.astype(np.float32),
        "phi":   phi.astype(np.float32),   "psi":   psi.astype(np.float32),
    }
