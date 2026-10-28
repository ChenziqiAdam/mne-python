"""Opt-in scientific runtime checks used by the SciBench pilot.

The module is inert unless ``SCIBENCH_TRIGGER_LOG`` names a log file. Checks
append stable IDs and never raise into MNE, never change a return value and
never touch global random state. Re-calling checkers work on copies and run with
checking suppressed (re-entrancy guard).

Tolerance convention (SANITIZER.md 5.8): ``_C * eps(dtype) * size * scale``
with each factor justified at the call site.

All imports of mne itself are lazy (inside functions) so that this module
sits outside the package import layering.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from __future__ import annotations

import contextlib
import functools
import json
import os
import sys
import warnings

import numpy as np

_ACTIVE = False
_EPS = float(np.finfo(np.float64).eps)
_C = 64.0
_ENV = "SCIBENCH_TRIGGER_LOG"
_MAX_ELEMS = 20_000_000
SWALLOWED = []
# the production hooks import this name (an explicit relative import, as the import-layering test requires)
_module = sys.modules[__name__]

REACHED = {}


def enabled():
    return bool(os.environ.get(_ENV)) and not _ACTIVE


def _eps_in(*objs):
    """Machine epsilon of the coarsest floating-point input (float64 default).

    The library computes in the precision of its inputs, so every law holds
    only to that precision (montage/layout coordinates, .fif operators and
    user arrays are often float32).
    """
    eps = _EPS
    for o in objs:
        try:
            if o is None:
                continue
            if hasattr(o, "data") and hasattr(o, "info"):  # Evoked / Raw
                o = o.data
            elif isinstance(o, dict) and "data" in o and not hasattr(o["data"], "keys"):
                o = o["data"]
            if isinstance(o, (list, tuple)):
                eps = max(eps, _eps_in(*o))
                continue
            dt = np.asarray(o).dtype if not hasattr(o, "dtype") else o.dtype
            if dt.kind == "c":
                dt = np.dtype("f%d" % (dt.itemsize // 2))
            if dt.kind == "f":
                eps = max(eps, float(np.finfo(dt).eps))
        except Exception:
            continue
    return eps


def trigger(checker_id):
    path = os.environ.get(_ENV)
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"checker_id": checker_id}) + "\n")
    except Exception:
        _swallowed()


def trigger_if(condition, checker_id):
    _reach(checker_id)
    if bool(condition):
        trigger(checker_id)


def _reach(checker_id):
    """Curator-only: count predicate evaluations (observation reachability)."""
    if os.environ.get("SCIBENCH_CHECKER_DEBUG"):
        REACHED[checker_id] = REACHED.get(checker_id, 0) + 1


def _swallowed():
    """Curator-only: record exceptions swallowed inside a checker."""
    if os.environ.get("SCIBENCH_CHECKER_DEBUG"):
        import traceback

        SWALLOWED.append(traceback.format_exc())


def _guarded(fn):
    """Run ``fn`` once, never nested, never raising into MNE."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        global _ACTIVE
        if _ACTIVE or not os.environ.get(_ENV):
            return None
        _ACTIVE = True
        try:
            # the host test suite may turn warnings into errors; a checker must
            # never be aborted by one
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with np.errstate(all="ignore"), _quiet():
                    return fn(*args, **kwargs)
        except Exception:
            _swallowed()
            return None
        finally:
            _ACTIVE = False

    return wrapper


@contextlib.contextmanager
def _quiet():
    from .utils import use_log_level

    with use_log_level("ERROR"):
        yield


def snap(x):
    """Copy of an array for a later re-call (None if too large or not numeric)."""
    try:
        x = np.asarray(x)
        if x.dtype.kind not in "fc" or x.size > _MAX_ELEMS:
            return None
        return x.copy()
    except Exception:
        return None


def resample_call(x, up, down, axis, window, pad, npad, method):
    x = snap(x)
    if x is None:
        return None
    return dict(x=x, up=up, down=down, axis=axis, window=window, pad=pad,
                npad=npad, method=method)


def psd_call(x, sfreq, fmin, fmax, bandwidth, adaptive, low_bias, normalization,
             remove_dc, output, max_iter):
    x = snap(x)
    if x is None:
        return None
    return dict(x=x, sfreq=sfreq, fmin=fmin, fmax=fmax, bandwidth=bandwidth,
                adaptive=adaptive, low_bias=low_bias, normalization=normalization,
                remove_dc=remove_dc, output=output, max_iter=max_iter)


def _finite(*arrs):
    return all(np.isfinite(np.asarray(a)).all() for a in arrs)


def _rot_defect(R):
    """Max |R^T R - I| and |det R - 1| over a stack of 3x3 matrices."""
    R = np.asarray(R, float)
    RtR = np.einsum("...ji,...jk->...ik", R, R)
    orth = np.abs(RtR - np.eye(3)).max(axis=(-2, -1))
    det = np.abs(np.linalg.det(R) - 1.0)
    return orth, det


# ---------------------------------------------------------------------------
# A. Coordinate transforms
# ---------------------------------------------------------------------------


@_guarded
def check_rotation3d(R, angles):
    """MNE-TRN-001."""
    # cos/sin are evaluated in the precision numpy promotes the angles to (integer
    # angles become float16/float32), whatever the dtype of the result
    eff = [np.zeros(1, np.cos(np.zeros((), np.asarray(a).dtype)).dtype) for a in angles]
    eps = _eps_in(R, *eff)
    R = np.asarray(R)
    if R.shape != (3, 3) or not _finite(R):
        return
    # entries are bounded by 1 and are sums of <= 3 products of unit-bounded
    # factors, so C * eps is a safe orthogonality bound
    tol = _C * eps
    orth, det = _rot_defect(R)
    trigger_if(orth > tol or det > tol, "MNE-TRN-001")


@_guarded
def check_quat_to_rot(quat, rot):
    """MNE-TRN-002."""
    eps = _eps_in(quat, rot)
    quat = np.asarray(quat, float)
    rot = np.asarray(rot, float)
    if quat.shape[-1] != 3 or rot.shape != quat.shape[:-1] + (3, 3):
        return
    if not _finite(quat, rot):
        return
    q2 = (quat * quat).sum(-1)
    ok = q2 <= 1.0  # documented unit-quaternion vector part
    if not ok.any():
        return
    # the real part is sqrt(1 - |q|^2): conditioning 1 / w
    w = np.sqrt(np.maximum(1.0 - q2, 0.0))
    tol = _C * eps / np.maximum(w, 1e-3)
    orth, det = _rot_defect(rot)
    bad = ok & ((orth > tol) | (det > tol))
    trigger_if(bad.any(), "MNE-TRN-002")


@_guarded
def check_rot_to_quat(rot, quat):
    """MNE-TRN-003."""
    eps = _eps_in(rot, quat)
    from .transforms import quat_to_rot

    rot = np.asarray(rot, float)
    quat = np.asarray(quat, float)
    if rot.shape[-2:] != (3, 3) or quat.shape != rot.shape[:-2] + (3,):
        return
    if not _finite(rot, quat):
        return
    orth, det = _rot_defect(rot)
    w = np.sqrt(np.maximum(1.0 + np.trace(rot, axis1=-2, axis2=-1), 0.0)) / 2.0
    ok = (orth <= 1e-6) & (np.abs(det) < 2) & (np.linalg.det(rot) > 0) & (w >= 1e-2)
    if not ok.any():
        return
    back = quat_to_rot(quat)
    diff = np.abs(back - rot).max(axis=(-2, -1))
    # extraction from a matrix that is itself off-orthogonal by `orth` (e.g. read back as float32)
    # is amplified by the same conditioning; the 3-component quaternion rebuilds `w` from `1 - |q|^2`, whose
    # rounding (eps/w^2 on |q|, hence eps/w^3 on `w`) dominates the round trip near a half turn
    tol = (_C * eps + 2.0 * orth) / w**3
    trigger_if((ok & (diff > tol)).any(), "MNE-TRN-003")


@_guarded
def check_ras_to_neuromag(nasion, lpa, rpa, trans):
    """MNE-TRN-004."""
    # the transform is built in the precision of the inputs (montage
    # fiducials are float32), so every tolerance scales with that epsilon
    eps = max(
        _EPS,
        *(
            float(np.finfo(np.asarray(p).dtype).eps)
            for p in (nasion, lpa, rpa)
            if np.asarray(p).dtype.kind == "f"
        ),
    )
    nasion, lpa, rpa = (np.asarray(p, float) for p in (nasion, lpa, rpa))
    trans = np.asarray(trans, float)
    if trans.shape != (4, 4) or not _finite(nasion, lpa, rpa, trans):
        return
    s = float(np.abs(np.concatenate([nasion, lpa, rpa])).max())
    right = rpa - lpa
    nr = np.linalg.norm(right)
    if s == 0 or nr == 0:
        return
    ru = right / nr
    origin = lpa + np.dot(nasion - lpa, ru) * ru
    d = np.linalg.norm(nasion - origin)
    if d < 1e-6 * s:
        return
    R, t = trans[:3, :3], trans[:3, 3]
    ml, mr, mn = (R @ p + t for p in (lpa, rpa, nasion))
    # the anterior axis is orthogonal to the left-right axis only to
    # eps * s / d (the origin is a cancellation), and that error multiplies
    # coordinates of size <= s
    tol0 = _C * eps * s * max(1.0, s / d)
    tol_o = _C * eps * max(1.0, s / d)
    orth, det = _rot_defect(R)
    zero_bad = (
        abs(ml[1]) > tol0
        or abs(ml[2]) > tol0
        or abs(mr[1]) > tol0
        or abs(mr[2]) > tol0
        or abs(mn[0]) > tol0
        or abs(mn[2]) > tol0
    )
    order_bad = not (ml[0] < mr[0]) or not (mn[1] > 0)
    rigid_bad = orth > tol_o or det > tol_o
    trigger_if(zero_bad or order_bad or rigid_bad, "MNE-TRN-004")


@_guarded
def check_find_vector_rotation(a, b, R):
    """MNE-TRN-005."""
    eps = _eps_in(a, b, R)
    a, b, R = np.asarray(a, float), np.asarray(b, float), np.asarray(R, float)
    if a.shape != (3,) or b.shape[-1:] != (3,) or R.shape != b.shape[:-1] + (3, 3):
        return
    if not _finite(a, b, R):
        return
    slack = abs(np.linalg.norm(a) - 1.0) + np.abs(np.linalg.norm(b, axis=-1) - 1.0)
    unit = slack <= 2e-12
    sin = np.linalg.norm(np.cross(a, b), axis=-1)
    ok = unit & (sin >= 1e-8)
    if not ok.any():
        return
    # the rotation axis a x b has absolute error eps; its direction error is
    # amplified by 1 / sin(theta)
    # (the formula assumes unit vectors: a norm slack `slack` in the inputs shifts the result by ~2 slack)
    tol = _C * (eps + slack) / np.where(ok, sin, 1.0)
    orth, det = _rot_defect(R)
    miss = np.abs(np.einsum("...ij,j->...i", R, a) - b).max(axis=-1)
    bad = ok & ((miss > tol) | (orth > tol) | (det > tol))
    trigger_if(bad.any(), "MNE-TRN-005")


@_guarded
def check_cart_to_sph(cart, sph):
    """MNE-TRN-006."""
    from .transforms import _sph_to_cart

    # float32 inputs (layouts, montages) carry float32 rounding through the trip
    eps = max(
        _EPS,
        *(
            float(np.finfo(np.asarray(a).dtype).eps)
            for a in (cart, sph)
            if np.asarray(a).dtype.kind == "f"
        ),
    )
    cart, sph = np.asarray(cart, float), np.asarray(sph, float)
    if cart.ndim != 2 or cart.shape[1] != 3 or sph.shape != cart.shape:
        return
    if not _finite(cart, sph):
        return
    r = np.linalg.norm(cart, axis=1)
    ok = (r >= 1e-100) & (r <= 1e100)
    if not ok.any():
        return
    back = _sph_to_cart(sph)
    err = np.linalg.norm(back - cart, axis=1)
    # polar angle from arccos(z / r): the round-trip error is r * eps / sin(theta)
    # (conditioning 1 / sin(theta)), capped at sqrt(2 eps) at a pole
    sin_t = np.hypot(cart[:, 0], cart[:, 1]) / np.where(ok, r, 1.0)
    tol = r * np.minimum(
        8.0 * np.sqrt(eps), 4.0 * _C * eps / np.maximum(sin_t, 1e-300)
    )
    trigger_if((ok & (err > tol)).any(), "MNE-TRN-006")


@_guarded
def check_average_quats(quats, weights, avg):
    """MNE-TRN-007."""
    eps = _eps_in(quats, avg)
    from .transforms import (
        _average_quats,
        _quat_real,
        quat_to_rot,
        rot_to_quat,
        rotation3d,
    )

    quats = np.asarray(quats, float)
    if quats.ndim != 2 or quats.shape[1] != 3 or len(quats) < 2:
        return
    weights = np.ones(len(quats)) if weights is None else np.asarray(weights, float)
    if not _finite(quats, weights) or (weights < 0).any() or weights.sum() <= 0:
        return
    if ((quats * quats).sum(-1) > 1.0).any():
        return
    w = weights / weights.sum()
    R_in = quat_to_rot(quats)
    R_Q = rotation3d(0.31, -0.52, 0.77)  # fixed generic rotation
    q_rot = rot_to_quat(R_Q @ R_in)
    w_min = min(_quat_real(quats).min(), _quat_real(q_rot).min())
    if w_min < 1e-2:
        return
    q4 = np.concatenate([_quat_real(quats)[:, None], quats], axis=-1)
    A = np.einsum("i,ij,ik->jk", w * w, q4, q4)  # the implementation scales q (not q q^T) by the weight
    ev = np.linalg.eigvalsh(A)
    gap = ev[-1] - ev[-2]
    if gap < 1e-6 * np.trace(A):
        return
    avg2 = _average_quats(q_rot, weights)
    expect = R_Q @ quat_to_rot(np.asarray(avg, float))
    diff = np.abs(quat_to_rot(avg2) - expect).max()
    # eigenvector perturbation ~ ||dA|| / gap, and A is a sum of n terms
    # (rounding bound n * eps); the real parts are rebuilt from `1 - |q|^2` (eps/w^3, see TRN-003)
    trigger_if(diff > _C * eps * len(quats) / (gap * w_min**3), "MNE-TRN-007")


@_guarded
def check_quat_mult(one, two, out):
    """MNE-TRN-008."""
    eps = _eps_in(one, two, out)
    from .transforms import _quat_real, quat_to_rot

    one, two, out = (np.asarray(q, float) for q in (one, two, out))
    if one.shape[-1] != 3 or two.shape[-1] != 3 or out.shape[-1] != 3:
        return
    if not _finite(one, two, out):
        return
    w1, w2 = _quat_real(one), _quat_real(two)
    w3 = np.abs(
        w1 * w2 - one[..., 0] * two[..., 0] - one[..., 1] * two[..., 1]
        - one[..., 2] * two[..., 2]
    )
    n1, n2 = (one * one).sum(-1), (two * two).sum(-1)
    wmin = np.minimum(np.minimum(w1, w2), w3)
    ok = (wmin >= 1e-4) & (n1 <= 1.0) & (n2 <= 1.0)
    if not ok.any():
        return
    lhs = quat_to_rot(out)
    rhs = quat_to_rot(one) @ quat_to_rot(two)
    diff = np.abs(lhs - rhs).max(axis=(-2, -1))
    tol = _C * eps / np.maximum(wmin, 1e-4)
    trigger_if((ok & (diff > tol)).any(), "MNE-TRN-008")


# ---------------------------------------------------------------------------
# B. Multiple-comparison and permutation statistics
# ---------------------------------------------------------------------------

_BH = ("i", "indep", "p", "poscorr")
_BY = ("n", "negcorr")


def _pvals_ok(p):
    p = np.asarray(p, float)
    return p.size > 0 and _finite(p) and (p >= 0).all() and (p <= 1).all()


@_guarded
def check_fdr(pvals, alpha, method, reject, adj):
    """MNE-STAT-001 / 002 / 004."""
    eps = _eps_in(pvals, adj)
    from .stats import fdr_correction

    p = np.asarray(pvals, float)
    reject, adj = np.asarray(reject, bool), np.asarray(adj, float)
    if not _pvals_ok(p) or not (0 < alpha < 1) or adj.shape != p.shape:
        return
    # STAT-001: the rejection set is {adjusted p < alpha}
    t = _C * eps * alpha
    bad1 = (reject & (adj > alpha + t)) | (~reject & (adj < alpha - t))
    trigger_if(bad1.any(), "MNE-STAT-001")
    # STAT-002: adjusted p dominates the raw p, is <= 1 and keeps the order
    pf, af = p.ravel(), adj.ravel()
    tol = _C * eps * np.maximum(pf, af)
    order = np.argsort(pf, kind="stable")
    a_sorted, tol_sorted = af[order], tol[order]
    bad2 = (
        (af < pf - tol).any()
        or (af > 1 + _C * eps).any()
        or (np.diff(a_sorted) < -np.maximum(tol_sorted[1:], tol_sorted[:-1])).any()
    )
    trigger_if(bad2, "MNE-STAT-002")
    # STAT-004: Benjamini-Yekutieli is no less conservative than BH
    if method in _BH:
        rej_bh, adj_bh, rej_by, adj_by = reject, adj, *fdr_correction(p, alpha, "negcorr")
    elif method in _BY:
        rej_by, adj_by, rej_bh, adj_bh = reject, adj, *fdr_correction(p, alpha, "indep")
    else:
        return
    tol4 = _C * eps * np.maximum(adj_bh, adj_by)
    bad4 = (rej_by & ~rej_bh & (adj_by < alpha - t)).any() or (
        adj_by < adj_bh - tol4
    ).any()
    trigger_if(bad4, "MNE-STAT-004")


@_guarded
def check_bonferroni(pval, alpha, reject, adj):
    """MNE-STAT-003."""
    eps = _eps_in(pval, adj)
    from .stats import fdr_correction

    p = np.asarray(pval, float)
    reject, adj = np.asarray(reject, bool), np.asarray(adj, float)
    if not _pvals_ok(p) or not (0 < alpha < 1) or adj.shape != p.shape:
        return
    rej_bh, adj_bh = fdr_correction(p, alpha, "indep")
    t = _C * eps * alpha
    tol = _C * eps * np.maximum(adj, adj_bh)
    bad = (reject & ~rej_bh & (adj < alpha - t)).any() or (adj < adj_bh - tol).any()
    trigger_if(bad, "MNE-STAT-003")


@_guarded
def check_pval_histogram(H0, pval):
    """MNE-STAT-005."""
    eps = _eps_in(H0, pval)
    H0, pval = np.asarray(H0, float), np.asarray(pval, float)
    if H0.size == 0 or not _finite(H0) or pval.size == 0 or not _finite(pval):
        return
    trigger_if((pval < 1.0 / H0.size - eps).any(), "MNE-STAT-005")


# ---------------------------------------------------------------------------
# C. Filtering and resampling
# ---------------------------------------------------------------------------


@_guarded
def check_fir_symmetry(h, phase):
    """MNE-FLT-001."""
    eps = _eps_in(h)
    if phase not in ("linear", "zero", "zero-double"):
        return
    h = np.asarray(h, float)
    if h.ndim != 1 or h.size < 2 or not _finite(h):
        return
    n = h.size
    # FFT-based design (firwin2) is symmetric to N * eps relative to the peak
    tol = _C * eps * n * np.abs(h).max()
    trigger_if(np.abs(h - h[::-1]).max() > tol, "MNE-FLT-001")


_SYM_PADS = ("reflect_limited", "reflect", "edge", "constant")


@_guarded
def check_overlap_add(x_in, h, n_fft, phase, picks, pad, out):
    """MNE-FLT-002 / 003."""
    eps = _eps_in(x_in, h, out)
    from .filter import _overlap_add_filter

    if x_in is None or phase != "zero" or pad not in _SYM_PADS:
        return
    h = np.asarray(h, float)
    out = np.asarray(out)
    if h.ndim != 1 or len(h) < 3 or out.shape != x_in.shape or not _finite(x_in, h):
        return
    if x_in.dtype.kind != "f" or x_in.size * 3 > _MAX_ELEMS:
        return
    n_h = len(h)
    # FLT-001 tolerance: the kernel must be symmetric for the law to apply
    if np.abs(h - h[::-1]).max() > _C * eps * n_h * np.abs(h).max():
        return

    def run(x):
        return _overlap_add_filter(
            x.copy(), h, n_fft, "zero", picks, 1, True, pad
        )

    x = x_in.reshape(-1, x_in.shape[-1]) if x_in.ndim != 2 else x_in
    n_t = x.shape[-1]
    n_fft_eff = n_fft if n_fft is not None else max(2 * n_h - 1, 2)
    logn = np.log2(max(n_fft_eff, 2 * n_h, n_t, 4))
    eps_out = float(np.finfo(out.dtype).eps) if out.dtype.kind == "f" else eps
    scale_h = np.linalg.norm(h)

    def l2(a):
        return np.linalg.norm(np.asarray(a, float).reshape(-1, a.shape[-1]), axis=-1)

    # FLT-002: reversal symmetry
    rev = run(x_in[..., ::-1])
    diff = l2(rev - out[..., ::-1])
    # the kernel asymmetry admitted by the precondition acts as a filter `h - h[::-1]`
    asym_gain = np.abs(np.fft.rfft(h - h[::-1], n=4 * n_h)).max()
    tol = _C * (
        eps * logn * l2(x_in) * scale_h + eps_out * (l2(rev) + l2(out))
    ) + 1.05 * asym_gain * l2(x_in)
    trigger_if((diff > tol).any(), "MNE-FLT-002")
    # FLT-003: additivity with y = reversed x
    y = x_in[..., ::-1]
    o_y = rev  # filt(y) with y the reversed x
    o_sum = run(x_in + y)
    diff3 = l2(o_sum - out - o_y)
    tol3 = _C * (
        eps * logn * (l2(x_in) + l2(y)) * scale_h
        + eps_out * (l2(o_sum) + l2(out) + l2(o_y))
    )
    trigger_if((diff3 > tol3).any(), "MNE-FLT-003")


@_guarded
def check_resample(call, y):
    """MNE-FLT-004."""
    eps = _eps_in(call["x"] if call else None, y)
    from .filter import resample

    if call is None:
        return
    x = call["x"]
    y = np.asarray(y)
    if x.dtype.kind != "f" or x.size * 3 > _MAX_ELEMS or not _finite(x, y):
        return
    kw = {k: v for k, v in call.items() if k != "x"}
    kw["n_jobs"] = 1
    rev = x[..., ::-1].copy()
    o_rev = resample(rev, **kw)
    o_sum = resample(x + rev, **kw)
    n = max(x.shape[kw["axis"]], o_sum.shape[kw["axis"]], 4)

    def l2(a):
        return np.linalg.norm(np.asarray(a, float))

    tol = _C * eps * np.log2(n) * (l2(y) + l2(o_rev) + l2(o_sum))
    trigger_if(l2(o_sum - y - o_rev) > tol, "MNE-FLT-004")


# ---------------------------------------------------------------------------
# D. Spectral estimation
# ---------------------------------------------------------------------------


@_guarded
def check_morlet(W, sigma_t, sfreq, f):
    """MNE-TF-001 (zero_mean=True wavelets only)."""
    W = np.asarray(W)
    # the oscillation must be sampled below the Nyquist frequency (aliased wavelets leak mass)
    if W.ndim != 1 or not _finite(W) or sigma_t * sfreq < 2.0 or not f < sfreq / 2.0:
        return
    # 5-sigma truncation and sampling leave < 1e-4 of the absolute mass
    trigger_if(abs(W.sum()) > 1e-4 * np.abs(W).sum(), "MNE-TF-001")


@_guarded
def check_psd_multitaper(call, psd):
    """MNE-TF-002."""
    eps = _eps_in(call["x"] if call else None, psd)
    from .time_frequency.multitaper import psd_array_multitaper

    if call is None or call["output"] != "power":
        return
    x = call["x"]
    psd = np.asarray(psd, float)
    ax = np.abs(x[x != 0])
    if x.dtype.kind != "f" or ax.size == 0 or ax.min() < 1e-100 or ax.max() > 1e100:
        return
    kw = {k: v for k, v in call.items() if k != "x"}
    psd2, _ = psd_array_multitaper(2.0 * x, n_jobs=1, **kw)
    # scaling by a power of two is exact in IEEE arithmetic, but batched FFTs
    # round differently, and a bin far below the spectral peak (leakage floor)
    # inherits rounding of order eps * peak, not eps * its own value
    peak = np.abs(psd).max(axis=-1, keepdims=True)
    tol = _C * eps * 4.0 * peak
    trigger_if((np.abs(psd2 - 4.0 * psd) > tol).any(), "MNE-TF-002")


@_guarded
def check_dpss(N, half_nbw, sym, low_bias, dpss, eigvals, fallback=False):
    """MNE-TF-003."""
    from scipy.linalg import toeplitz

    dpss, eigvals = np.asarray(dpss, float), np.asarray(eigvals, float)
    if not sym or not (2 <= N <= 2048) or dpss.ndim != 2 or dpss.shape[1] != N:
        return
    if not _finite(dpss, eigvals) or not (0 < half_nbw < N / 2):
        return
    W = half_nbw / N
    k = np.arange(N)
    col = np.empty(N)
    col[0] = 2.0 * W
    col[1:] = np.sin(2.0 * np.pi * W * k[1:]) / (np.pi * k[1:])
    B = toeplitz(col)
    conc = np.einsum("ki,ij,kj->k", dpss, B, dpss) / np.einsum("ki,ki->k", dpss, dpss)
    bad = (np.abs(conc - eigvals) > 1e-6).any()
    if low_bias and not fallback:  # the fallback (no taper above 0.9) is announced by the library's own warning
        bad = bad or (conc <= 0.9 - 1e-6).any()
    trigger_if(bad, "MNE-TF-003")


# ---------------------------------------------------------------------------
# E. Covariance
# ---------------------------------------------------------------------------


@_guarded
def check_covariance(cov, m2=None):
    """MNE-COV-001. ``m2``: per-channel uncentred second moment (the one-pass estimate subtracts it)."""
    eps = _eps_in(cov["data"])
    covs = cov if isinstance(cov, (list, tuple)) else [cov]
    bad = False
    for c in covs:
        data = np.asarray(c["data"])
        if data.ndim != 2 or data.shape[0] != data.shape[1] or data.shape[0] > 2000:
            continue
        if not _finite(data):
            continue
        n = data.shape[0]
        scale = np.abs(data).max()
        if scale == 0:
            continue
        diag = np.real(np.diag(data))
        # subtracting the mean from the uncentred second moment cancels `m2 / diag` of the digits
        amp = 1.0
        if m2 is not None and np.shape(m2) == (n,) and _finite(np.asarray(m2, float)):
            pos_d = diag > 0
            if pos_d.any():
                amp = max(1.0, float((np.abs(np.asarray(m2, float))[pos_d] / diag[pos_d]).max()))
        tol = _C * eps * scale * amp
        if (diag < -tol).any() or np.abs(data - data.conj().T).max() > tol:
            bad = True
            continue
        pos = diag > tol
        if pos.sum() < 2:
            continue
        # PSD is invariant under diagonal congruence; scale to unit diagonal so
        # channel types of very different magnitude (T, T/m, V) do not hide
        # one another below eps * lambda_max
        s = 1.0 / np.sqrt(diag[pos])
        M = (data[np.ix_(pos, pos)] * s[:, None]) * s[None, :]
        M = (M + M.conj().T) / 2
        lam = np.linalg.eigvalsh(M)
        bad = bad or lam[0] < -_C * n * eps * max(lam[-1], 1.0) * amp
    trigger_if(bad, "MNE-COV-001")


# ---------------------------------------------------------------------------
# F. Inverse operators and beamformers
# ---------------------------------------------------------------------------


@_guarded
def check_inverse_dspm(inv, K, noise_norm, method, pick_ori):
    """MNE-INV-001."""
    eps = _eps_in(K, inv["eigen_leads"], inv["eigen_fields"])
    from ._fiff.constants import FIFF

    if method != "dSPM" or pick_ori == "normal" or noise_norm is None:
        return
    S = np.asarray(inv["noise_cov"]["data"])
    if S.ndim != 2 or K.shape[1] != S.shape[0] or K.size > 5_000_000:
        return
    if not _finite(S, K, noise_norm):
        return
    Wp = np.asarray(inv["whitener"]) @ np.asarray(inv["proj"])
    # pca="white" keeps a zero row for each discarded eigenvalue
    Wp = Wp[np.linalg.norm(Wp, axis=1) > 0]
    r = Wp.shape[0]
    if r == 0 or np.abs(Wp @ S @ Wp.conj().T - np.eye(r)).max() > 1e-6:
        return  # the whitener is not whitening this noise model: law n/a
    eig = np.asarray(inv["noise_cov"]["eig"], float)
    eig = eig[eig > 0]
    if eig.size == 0:
        return
    kappa = eig.max() / eig.min()
    if kappa > 1e6:
        return
    var = np.einsum("ij,jk,ik->i", K, S, K).real
    if inv["source_ori"] == FIFF.FIFFV_MNE_FREE_ORI:
        if var.size % 3:
            return
        var = var.reshape(-1, 3).sum(axis=1)
    nn = np.asarray(noise_norm, float).reshape(-1)
    if nn.size != var.size:
        return
    tol = _C * eps * S.shape[0] * kappa
    trigger_if((np.abs(var * nn * nn - 1.0) > tol).any(), "MNE-INV-001")


@_guarded
def check_inverse_consistency(call, stc, method, pick_ori, free_ori):
    """MNE-INV-002 / 003."""
    from .minimum_norm.inverse import _apply_inverse

    if call is None or method not in ("MNE", "dSPM", "sLORETA"):
        return
    evoked = call["evoked"]
    if not _finite(evoked.data) or evoked.data.size > _MAX_ELEMS:
        return
    kw = {k: v for k, v in call.items() if k != "evoked"}
    kw["return_residual"] = False
    s = np.asarray(stc.data)
    if s.dtype.kind != "f":
        return
    # operators read from .fif store eigen_leads / eigen_fields as float32, so
    # the estimate is only reproducible to that precision
    eps = _EPS
    for key in ("eigen_leads", "eigen_fields"):
        try:
            dt = np.asarray(call["inverse_operator"][key]["data"]).dtype
        except Exception:
            continue
        if dt.kind == "f":
            eps = max(eps, float(np.finfo(dt).eps))
    # INV-003: linearity (scaling by 2 is exact)
    ev2 = evoked.copy()
    ev2.data = 2.0 * evoked.data
    s2 = np.asarray(_apply_inverse(ev2, **kw).data)
    tol = _C * eps * 2.0 * np.abs(s)
    trigger_if((np.abs(s2 - 2.0 * s) > tol).any(), "MNE-INV-003")
    # INV-002: magnitude equals the norm of the vector estimate
    if free_ori and pick_ori is None:
        kw_v = dict(kw)
        kw_v["pick_ori"] = "vector"
        v = np.asarray(_apply_inverse(evoked, **kw_v).data)
        mag = np.linalg.norm(v, axis=1)
        tol2 = _C * eps * max(np.abs(s).max(), 1e-300)
        trigger_if((np.abs(s - mag) > tol2).any(), "MNE-INV-002")


@_guarded
def check_beamformer(
    G, W, max_power_ori, n_orient_in, weight_norm, pick_ori, reduce_rank,
    inversion, orient_std, Cm_inv, call,
):
    """MNE-BF-001 (unit gain) and MNE-BF-002 (covariance-scale invariance)."""
    eps = _eps_in(G, W, Cm_inv)
    from .beamformer._compute_beamformer import _compute_beamformer

    if reduce_rank:
        return
    G, W = np.asarray(G), np.asarray(W)
    if G.size > _MAX_ELEMS:
        return
    n_ch = G.shape[0]
    n_src = G.shape[1] // n_orient_in
    Gk = np.reshape(G.T, (n_src, n_orient_in, n_ch)).transpose(0, 2, 1)
    if pick_ori == "normal":
        g = Gk[..., 2:3]
    elif pick_ori == "max-power":
        if max_power_ori is None:
            return
        g = np.matmul(Gk, np.asarray(max_power_ori)[..., None])
    else:
        g = Gk
        if n_orient_in > 1 and inversion != "matrix":
            return  # diagonal-only inversion: unit gain n/a
    n_out = g.shape[-1]
    if W.shape != (n_src * n_out, n_ch) or not _finite(G, W, Cm_inv):
        return
    Wk = W.reshape(n_src, n_out, n_ch)
    denom = np.matmul(np.matmul(g.swapaxes(-2, -1).conj(), Cm_inv), g)
    kappa = np.linalg.cond(denom)
    ok = np.isfinite(kappa) & (kappa <= 1e6)  # the library inverts via rcond=1e-7 truncation
    if weight_norm is None and np.all(np.asarray(orient_std) == 1) and ok.any():
        resid = np.abs(np.matmul(Wk, g) - np.eye(n_out)).max(axis=(-2, -1))
        # forming denom = g^H C^+ g rounds at eps * |g|^2 |C^+| (a sandwich
        # product, not eps * |denom|), and the solve amplifies that by kappa
        gn2 = np.linalg.norm(g, ord=2, axis=(-2, -1)) ** 2
        dn = np.linalg.norm(denom, ord=2, axis=(-2, -1))
        cn = np.linalg.norm(np.asarray(Cm_inv), ord=2)
        k_eff = np.where(ok, kappa * np.maximum(gn2 * cn / np.maximum(dn, 1e-300), 1.0), 1.0)
        tol = _C * eps * k_eff
        trigger_if((ok & (resid > tol)).any(), "MNE-BF-001")
    if weight_norm in (None, "unit-noise-gain", "unit-noise-gain-invariant"):
        sv = np.linalg.svd(np.asarray(Cm_inv), compute_uv=False)
        # rank-truncated pseudo-inverse: drop the (numerically) null directions
        sv = sv[sv > 1e-12 * sv.max()] if sv.size else sv
        kc = sv.max() / sv.min() if sv.size else np.inf
        if not np.isfinite(kc) or kc > 1e6 or kappa.max() > 1e6:
            return
        args = list(call)
        args[1] = 4.0 * np.asarray(args[1])
        W4, _ = _compute_beamformer(*args)
        tol = _C * eps * kc * kappa.max() * max(np.abs(W).max(), 1e-300)
        trigger_if(np.abs(np.asarray(W4) - W).max() > tol, "MNE-BF-002")


# ---------------------------------------------------------------------------
# G. Evoked data and forward model
# ---------------------------------------------------------------------------


@_guarded
def check_combine_evoked(all_evoked, weights, out):
    """MNE-EVK-001."""
    eps = _eps_in(all_evoked, out)
    from .evoked import combine_evoked

    if not isinstance(weights, str) or weights != "nave" or len(all_evoked) < 3:
        return
    if not all(_finite(e.data) for e in all_evoked):
        return
    head = combine_evoked(list(all_evoked[:2]), weights="nave")
    nested = combine_evoked([head, all_evoked[2], *all_evoked[3:]], weights="nave")
    # opposite-sign inputs cancel in the pool: rounding is set by the size of the terms, not of the result
    mag = max(np.abs(out.data).max(), *(np.abs(e.data).max() for e in all_evoked), 1e-300)
    d_tol = _C * eps * len(all_evoked) * mag
    bad = np.abs(nested.data - out.data).max() > d_tol
    bad = bad or abs(nested.nave - out.nave) > _C * eps * out.nave
    trigger_if(bad, "MNE-EVK-001")


@_guarded
def check_sphere_meg(rr, B, sphere):
    """MNE-FWD-001 (MEG gain of a spherical conductor)."""
    rr, B = np.asarray(rr, float), np.asarray(B, float)
    if rr.ndim != 2 or B.shape[0] != 3 * len(rr) or not _finite(rr, B):
        return
    r0 = np.asarray(sphere["r0"], float)
    rel = rr - r0
    dist = np.linalg.norm(rel, axis=1)
    ok = dist >= 1e-3
    if not ok.any():
        return
    Bk = B.reshape(len(rr), 3, -1)
    rhat = rel / np.where(ok, dist, 1.0)[:, None]
    # orthonormal tangent basis of each radial direction
    helper = np.where(
        (np.abs(rhat[:, 0]) < 0.9)[:, None], np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    )
    t1 = np.cross(rhat, helper)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(rhat, t1)
    rad = np.linalg.norm(np.einsum("nc,ncm->nm", rhat, Bk), axis=1)
    tan = np.maximum(
        np.linalg.norm(np.einsum("nc,ncm->nm", t1, Bk), axis=1),
        np.linalg.norm(np.einsum("nc,ncm->nm", t2, Bk), axis=1),
    )
    # cancellation amplification <= (r_sensor / r_dipole)^3 <= 8e6, times eps
    trigger_if((ok & (tan > 0) & (rad > 1e-8 * tan)).any(), "MNE-FWD-001")


# ---------------------------------------------------------------------------
# H. Projectors, ANOVA, filter design, interpolation, depth weighting
# ---------------------------------------------------------------------------


@_guarded
def check_projector(proj, nproj, U):
    """MNE-PRJ-001 / 002."""
    eps = _eps_in(proj, U)
    P, U = np.asarray(proj, float), np.asarray(U, float)
    n = P.shape[0]
    if P.ndim != 2 or P.shape[1] != n or nproj < 1 or nproj >= n or not _finite(P, U):
        return
    tol = _C * eps * n
    trigger_if(
        np.abs(P - P.T).max() > tol or np.abs(P @ P - P).max() > tol, "MNE-PRJ-001"
    )
    if U.shape != (n, nproj):
        trigger_if(True, "MNE-PRJ-002")
        return
    bad = (
        np.abs(U.T @ U - np.eye(nproj)).max() > tol
        or np.abs(P @ U).max() > tol
        or abs(np.trace(np.eye(n) - P) - nproj) > tol
    )
    trigger_if(bad, "MNE-PRJ-002")


@_guarded
def check_f_oneway(args, sigma, F):
    """MNE-STAT-006 / 007."""
    eps = _eps_in(args, F)
    from .stats import f_oneway
    from .stats.parametric import ttest_ind_no_p

    if sigma != 0 or len(args) < 2:
        return
    arrs = [np.asarray(a, float) for a in args]
    F = np.asarray(F, float)
    if any(a.shape[0] < 2 or a.shape[1:] != arrs[0].shape[1:] for a in arrs):
        return
    if not _finite(*arrs):
        return
    n_tot = sum(a.shape[0] for a in arrs)
    mu = np.concatenate(arrs, axis=0).mean(axis=0)
    cen = [a - mu for a in arrs]
    sswn = sum(((a - a.mean(axis=0)) ** 2).sum(axis=0) for a in cen)
    ss_all = sum((a**2).sum(axis=0) for a in arrs)
    ok = sswn > 0
    if not np.any(ok) or F.shape != sswn.shape:
        return
    ratio = np.where(ok, ss_all / np.where(ok, sswn, 1.0), 1.0)
    tol = _C * eps * n_tot * np.maximum(ratio, 1.0) * np.maximum(np.abs(F), 1.0)
    Fc = np.asarray(f_oneway(*cen))
    well = ok & np.isfinite(Fc) & (n_tot * ratio * _C * eps <= 1.0)  # cancellation not catastrophic
    wrong = (np.abs(F - Fc) > tol) | ~np.isfinite(F)
    trigger_if((well & wrong).any(), "MNE-STAT-006")
    if len(arrs) == 2:
        t = np.asarray(ttest_ind_no_p(arrs[0], arrs[1]))
        trigger_if((well & (np.abs(F - t * t) > tol)).any(), "MNE-STAT-007")


@_guarded
def check_fir_cutoff(h, freq, gain, sfreq, phase, fir_design):
    """MNE-FLT-005."""
    if phase != "zero" or fir_design != "firwin":
        return
    h = np.asarray(h, float)
    freq, gain = np.asarray(freq, float), np.asarray(gain, float)
    if h.ndim != 1 or len(h) < 5 or len(freq) != len(gain) or len(freq) < 3:
        return
    if not _finite(h, freq, gain) or not np.isin(gain, (0, 1)).all():
        return
    change = np.nonzero(gain[:-1] != gain[1:])[0]
    if len(change) != 1:
        return  # single low-/high-pass cutoff only
    fc = 0.5 * (freq[change[0]] + freq[change[0] + 1])
    w = 2 * np.pi * fc / sfreq
    k = np.arange(len(h)) - (len(h) - 1) / 2.0
    mag = abs(np.sum(h * np.exp(-1j * w * k)))
    trigger_if(abs(mag - 0.5) > 0.02, "MNE-FLT-005")


@_guarded
def check_hilbert(x, n_fft, envelope, out):
    """MNE-FLT-006."""
    eps = _eps_in(x, out)
    x, out = np.asarray(x), np.asarray(out)
    if x.dtype.kind != "f" or out.shape != x.shape or not _finite(x, out):
        return
    n = x.shape[-1]
    N = max(n_fft or n, n, 4)
    tol = _C * eps * np.log2(N) * np.linalg.norm(x, axis=-1, keepdims=True)
    if envelope:
        trigger_if(((np.abs(x) - out) > tol).any(), "MNE-FLT-006")
    else:
        trigger_if((np.abs(out.real - x) > tol).any(), "MNE-FLT-006")


@_guarded
def check_interpolation(C, alpha, interpolation):
    """MNE-INT-001."""
    eps = _eps_in(C)
    C, M = np.asarray(C, float), np.asarray(interpolation, float)
    a = 0.0 if alpha is None else alpha
    if not (0 <= a <= 1e-2) or not _finite(C, M):
        return
    kappa = np.linalg.cond(C)
    if not np.isfinite(kappa) or kappa > 1e10:
        return
    trigger_if(np.abs(M.sum(axis=1) - 1.0).max() > _C * eps * kappa, "MNE-INT-001")


@_guarded
def check_depth_prior(call, prior):
    """MNE-DEP-001."""
    eps = _eps_in(call["forward"]["sol"] if call else None, prior)
    import copy

    from .forward.forward import compute_depth_prior

    if call is None or call["limit"] is None:
        return
    fwd = call["forward"]
    prior = np.asarray(prior, float)
    if fwd["sol"]["data"].size > _MAX_ELEMS or not _finite(prior):
        return
    f2 = copy.deepcopy(fwd)
    f2["sol"]["data"] = 2.0 * f2["sol"]["data"]
    p2 = compute_depth_prior(f2, **{k: v for k, v in call.items() if k != "forward"})
    trigger_if(np.abs(p2 - prior).max() > _C * eps * prior.max(), "MNE-DEP-001")
