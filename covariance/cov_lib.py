#!/usr/bin/env python3
"""
cov_lib.py: covariance helpers shared by syst_covariance.py.

Not run directly. Provides the bin vector over (sample, variable, FC/PC)
blocks (BinVector), the universe covariance

    E^(d)_ij = 1/N_univ(d) * sum_alpha (N_i^alpha - N_i^ref) (N_j^alpha - N_j^ref),

correlation / fractional matrices, missing-bin handling, the matrix,
breakdown and error-band plots, and the .npz / .root writers.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from syst_lib import (  # noqa: E402
        VARIABLES, sample_mask, CONT_LABEL, short)
except ImportError as e:
    sys.exit(f"cov_lib.py needs syst_lib.py (and plot_cv_weights.py) in "
             f"the same folder: {e}")

SHORT_VAR = {
    "reco_E": r"$E_\nu^{\rm reco}$", "reco_Elep": r"$E_{\rm lep}^{\rm reco}$",
    "reco_Ehad": r"$E_{\rm had}^{\rm reco}$",
    "reco_cosz": r"$\cos\theta_z^{\rm reco}$",
    "true_E": r"$E_\nu^{\rm true}$", "true_Elep": r"$E_{\rm lep}^{\rm true}$",
    "true_Ehad": r"$E_{\rm had}^{\rm true}$",
    "true_cosz": r"$\cos\theta_z^{\rm true}$",
    "reco_y": r"$y^{\rm reco}$", "true_y": r"$y^{\rm true}$",
}
SHORT_SAMPLE = {"nue": r"$\nu_e$", "numu": r"$\nu_\mu$", "nc": "NC",
                "nue+numu": r"$\nu_e+\nu_\mu$",
                "all": r"$\nu_e+\nu_\mu$+NC"}


# --------------------------------------------------------------------------
# Binned "event-rate vector" over one or more (sample, variable) blocks
# --------------------------------------------------------------------------
class BinVector:
    """Maps events to a global bin index across blocks; fast re-histogramming.

    Bin convention matches np.histogram: [lo, hi) except the last bin, which
    includes its upper edge. Events outside the edges are not counted.
    """

    def __init__(self, blocks, v, keep):
        self.blocks = []            # dicts: sample, var, edges, cont, offset, nb
        self.ev = []                # per block: event indices
        self.ix = []                # per block: global bin index of each
        offset = 0
        for sample, var, edges, cont, cmask in blocks:
            sel = keep & cmask & sample_mask(sample, v["reco_pdg"])
            x = v[var]
            ev = np.nonzero(sel & np.isfinite(x))[0]
            xs = x[ev]
            nb = len(edges) - 1
            b = np.searchsorted(edges, xs, side="right") - 1
            b[xs == edges[-1]] = nb - 1
            ok = (b >= 0) & (b < nb)
            self.ev.append(ev[ok])
            self.ix.append(b[ok] + offset)
            self.blocks.append(dict(sample=sample, var=var, edges=edges,
                                    cont=cont, offset=offset, nb=nb))
            offset += nb
        self.n = offset

    def __call__(self, w):
        out = np.zeros(self.n)
        for ev, ix in zip(self.ev, self.ix):
            out += np.bincount(ix, weights=w[ev], minlength=self.n)
        return out


def cov_from_universes(n_cv, n_univ):
    """E = 1/N_univ sum_alpha (N^alpha - N^CV)(N^alpha - N^CV)^T."""
    d = np.asarray(n_univ) - n_cv[None, :]
    return d.T @ d / len(n_univ)


def missing_bins(E):
    """Bins with a nonfinite variance (a nonfinite stored response touched
    them); their rows/columns stay NaN (shown blank) instead of 0."""
    return ~np.isfinite(np.diag(E))


def correlation(E):
    s = np.sqrt(np.clip(np.diag(E), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = E / np.outer(s, s)
    r[~np.isfinite(r)] = 0.0          # zero-variance bins
    miss = missing_bins(E)
    r[miss, :] = np.nan
    r[:, miss] = np.nan
    return r


def fractional(E, n_cv):
    with np.errstate(divide="ignore", invalid="ignore"):
        f = E / np.outer(n_cv, n_cv)
    f[~np.isfinite(f)] = 0.0          # empty bins
    miss = missing_bins(E) | ~np.isfinite(n_cv)
    f[miss, :] = np.nan
    f[:, miss] = np.nan
    return f


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def axis_coords(bv, args):
    """Bin-edge coordinates used for drawing.

    Returns (coords, true_edges_or_None, is_log).
    * --logx on a single block of a positive variable: physical edges on a
      log axis, so bin widths keep their true relative sizes. A lower edge
      of 0 cannot sit on a log axis; it is drawn at --logx-zero (default:
      one more bin of the same log-width as the first non-zero bin) and
      still labelled "0".
    * otherwise: bin index 0..n (every bin the same width).
    """
    if args.logx and len(bv.blocks) == 1:
        e = np.asarray(bv.blocks[0]["edges"], float)
        if e[0] < 0:
            return np.arange(bv.n + 1, dtype=float), None, False
        c = e.copy()
        if c[0] == 0:
            c[0] = (args.logx_zero if args.logx_zero
                    else (e[1] ** 2 / e[2] if len(e) > 2 else e[1] / 10))
        return c, e, True
    return np.arange(bv.n + 1, dtype=float), None, False


def _log_ticks(ax, coords, edges, args, axis="x"):
    from matplotlib.ticker import FixedLocator, NullLocator
    labels = [f"{x:g}" for x in edges]
    loc = FixedLocator(coords)
    if axis == "x":
        ax.set_xscale("log")
        ax.set_xlim(coords[0], coords[-1])
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels(labels, rotation=45 if len(edges) > 8 else 0,
                           fontsize=args.tick_fontsize)
    else:
        ax.set_yscale("log")
        ax.set_ylim(coords[0], coords[-1])
        ax.yaxis.set_major_locator(loc)
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_yticklabels(labels, fontsize=args.tick_fontsize)


def _axes_ticks(ax, bv, args):
    """Block separators + labels, or bin-edge labels for a single block."""
    n = bv.n
    coords, edges, is_log = axis_coords(bv, args)
    if is_log:
        _log_ticks(ax, coords, edges, args, "x")
        _log_ticks(ax, coords, edges, args, "y")
        lab = VARIABLES[bv.blocks[0]["var"]][0]
        ax.set_xlabel(lab, fontsize=args.label_fontsize)
        ax.set_ylabel(lab, fontsize=args.label_fontsize)
        return
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    if len(bv.blocks) == 1:
        e = bv.blocks[0]["edges"]
        ticks = np.arange(len(e))
        labels = [f"{x:g}" for x in e]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=45 if len(e) > 8 else 0,
                           fontsize=args.tick_fontsize)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=args.tick_fontsize)
        lab = VARIABLES[bv.blocks[0]["var"]][0]
        ax.set_xlabel(lab, fontsize=args.label_fontsize)
        ax.set_ylabel(lab, fontsize=args.label_fontsize)
    else:
        mids, labels = [], []
        for b in bv.blocks:
            if b["offset"] > 0:
                ax.axvline(b["offset"], color="k", lw=1)
                ax.axhline(b["offset"], color="k", lw=1)
            mids.append(b["offset"] + b["nb"] / 2)
            labels.append(f"{CONT_LABEL[b['cont']]}{SHORT_SAMPLE[b['sample']]} "
                          f"{SHORT_VAR[b['var']]}")
        ax.set_xticks(mids)
        ax.set_xticklabels(labels, rotation=45, ha="right",
                           fontsize=args.tick_fontsize)
        ax.set_yticks(mids)
        ax.set_yticklabels(labels, fontsize=args.tick_fontsize)
        ax.set_xlabel("Bin", fontsize=args.label_fontsize)
        ax.set_ylabel("Bin", fontsize=args.label_fontsize)


def universe_curves(spectra, sigs, label_prefix="", color=None, lw=1.5):
    """Universe spectra -> curve dicts. + sigma solid, - sigma dashed."""
    out = []
    for h, sg in sorted(zip(spectra, sigs), key=lambda t: (abs(t[1]), -t[1])):
        k = int(round(abs(sg)))
        out.append(dict(h=h, ls="-" if sg > 0 else "--", lw=lw,
                        color=color or SIGMA_COLORS.get(k, "tab:gray"),
                        label=f"{label_prefix}{sg:+g}$\\sigma$"))
    return out


def plot_matrix(M, bv, title, cbar_label, path, args, kind):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=args.figsize)
    if kind == "corr":
        vmin, vmax, cmap = -1, 1, args.cmap
    else:
        m = np.nanmax(np.abs(M)) or 1.0
        vmin, vmax, cmap = -m, m, args.cmap
    edges, _, is_log = axis_coords(bv, args)
    pc = ax.pcolormesh(edges, edges, M, cmap=cmap, vmin=vmin, vmax=vmax,
                       shading="flat")
    cb = fig.colorbar(pc, ax=ax)
    cb.set_label(cbar_label, fontsize=args.label_fontsize)
    _axes_ticks(ax, bv, args)
    ax.set_aspect("equal" if not is_log else "auto")
    if is_log:
        ax.set_box_aspect(1)
    ax.set_title(title, fontsize=args.title_fontsize)
    if args.annotate and bv.n <= args.annotate_max:
        mid = (np.sqrt(edges[:-1] * edges[1:]) if is_log
               else 0.5 * (edges[:-1] + edges[1:]))
        for i in range(bv.n):
            for j in range(bv.n):
                ax.text(mid[j], mid[i], f"{M[i, j]:.2g}", ha="center",
                        va="center", fontsize=max(4, 9 - bv.n // 4))
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def plot_breakdown(n_cv, E_tot, E_dials, bv, title, path, args):
    """Fractional uncertainty per bin: total + the largest dial contributions."""
    import matplotlib.pyplot as plt
    with np.errstate(divide="ignore", invalid="ignore"):
        tot = np.where(n_cv > 0, np.sqrt(np.clip(np.diag(E_tot), 0, None))
                       / n_cv, np.nan)
        per = {d: np.where(n_cv > 0, np.sqrt(np.clip(np.diag(E), 0, None))
                           / n_cv, np.nan) for d, E in E_dials.items()}
    rank = sorted(per, key=lambda d: -np.nansum(per[d] ** 2 * n_cv ** 2))
    top = rank[:args.top_dials]

    fig, ax = plt.subplots(figsize=(max(8, bv.n * 0.18 + 4), 5))
    x, _, _ = axis_coords(bv, args)
    cmap = plt.get_cmap("tab10" if len(top) <= 10 else "tab20")
    for k, d in enumerate(top):
        ax.stairs(per[d], x, baseline=None, color=cmap(k % cmap.N), lw=1.3,
                  label=short(d, args.short_labels))
    if len(rank) > len(top):
        rest = np.zeros(bv.n)
        for d in rank[len(top):]:
            rest += np.diag(E_dials[d])
        with np.errstate(divide="ignore", invalid="ignore"):
            rest = np.where(n_cv > 0, np.sqrt(rest) / n_cv, np.nan)
        ax.stairs(rest, x, baseline=None, color="gray", ls="--", lw=1.2,
                  label=f"other {len(rank) - len(top)} dials")
    ax.stairs(tot, x, baseline=None, color="black", lw=2.2, label="Total")
    _axes_ticks_1d(ax, bv, args)
    ax.set_ylabel(r"Fractional uncertainty $\sqrt{E_{ii}}/N_i$",
                  fontsize=args.label_fontsize)
    ax.set_ylim(bottom=0)
    ax.set_title(title, fontsize=args.title_fontsize)
    ax.legend(fontsize=args.legend_fontsize, ncol=args.legend_ncol,
              loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


SIGMA_COLORS = {1: "tab:red", 2: "tab:orange", 3: "tab:green"}


def plot_band(n_cv, E_tot, bv, title, path, args,
              band_label=r"$\pm\sqrt{E_{ii}}$ (sum of dials)",
              curves=(), cv_label="CV"):
    """CV spectrum with the error band sigma_i = sqrt(E_ii), plus a ratio
    panel showing 1 +/- sigma_i / N_i.

    curves: iterable of dicts {h, label, color, ls} -- the universe spectra
    (e.g. the dial at cv -1 sigma and cv +1 sigma) drawn on top of the band.
    """
    import matplotlib.pyplot as plt
    err = np.sqrt(np.clip(np.diag(E_tot), 0, None))
    fig, (ax, axr) = plt.subplots(
        2, 1, sharex=True, figsize=(max(8, bv.n * 0.18 + 4), 6.5),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})
    x, _, _ = axis_coords(bv, args)
    if args.per_width:      # display only: divide each bin by its width
        width = np.concatenate([np.diff(b["edges"]) for b in bv.blocks])
        n_cv, err = n_cv / width, err / width
        curves = [dict(c, h=c["h"] / width) for c in curves]
    ax.stairs(n_cv + err, x, baseline=n_cv - err, fill=True, color="tab:blue",
              alpha=0.3, label=band_label)
    ax.stairs(n_cv, x, color="black", lw=2, label=cv_label)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(n_cv > 0, err / n_cv, np.nan)
        ratios = [np.where(n_cv > 0, c["h"] / n_cv, np.nan) for c in curves]
    for c, r in zip(curves, ratios):
        ax.stairs(c["h"], x, baseline=None, color=c["color"], ls=c["ls"],
                  lw=c.get("lw", 1.5), label=c["label"])
        axr.stairs(r, x, baseline=None, color=c["color"], ls=c["ls"],
                   lw=c.get("lw", 1.5))
    axr.stairs(1 + rel, x, baseline=1 - rel, fill=True, color="tab:blue",
               alpha=0.3)
    axr.axhline(1, color="black", lw=1)
    vals = [1 + rel, 1 - rel] + ratios
    vals = np.concatenate([v[np.isfinite(v)] for v in vals])
    if len(vals):
        lo, hi = min(vals.min(), 1.0), max(vals.max(), 1.0)
        span = max(hi - lo, 0.02)
        axr.set_ylim(max(lo - 0.12 * span, 0), hi + 0.12 * span)
    axr.set_ylabel("Ratio to CV", fontsize=args.label_fontsize * 0.85)
    _axes_ticks_1d(axr, bv, args)
    ax.tick_params(labelbottom=False)
    ax.set_xlabel("")
    ax.set_ylabel(args.ylabel or ("Events / bin width" if args.per_width
                                  else "Events"),
                  fontsize=args.label_fontsize)
    if args.logy:
        ax.set_yscale("log")
    ax.set_title(title, fontsize=args.title_fontsize)
    ax.legend(fontsize=args.legend_fontsize, ncol=args.legend_ncol)
    fig.align_ylabels([ax, axr])
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def _axes_ticks_1d(ax, bv, args):
    coords, edges, is_log = axis_coords(bv, args)
    if is_log:
        _log_ticks(ax, coords, edges, args, "x")
        ax.set_xlabel(VARIABLES[bv.blocks[0]["var"]][0],
                      fontsize=args.label_fontsize)
        return
    ax.set_xlim(0, bv.n)
    if len(bv.blocks) == 1:
        e = bv.blocks[0]["edges"]
        ax.set_xticks(np.arange(len(e)))
        ax.set_xticklabels([f"{x:g}" for x in e],
                           rotation=45 if len(e) > 8 else 0,
                           fontsize=args.tick_fontsize)
        ax.set_xlabel(VARIABLES[bv.blocks[0]["var"]][0],
                      fontsize=args.label_fontsize)
    else:
        mids, labels = [], []
        for b in bv.blocks:
            if b["offset"] > 0:
                ax.axvline(b["offset"], color="k", lw=1)
            mids.append(b["offset"] + b["nb"] / 2)
            labels.append(f"{CONT_LABEL[b['cont']]}{SHORT_SAMPLE[b['sample']]} "
                          f"{SHORT_VAR[b['var']]}")
        ax.set_xticks(mids)
        ax.set_xticklabels(labels, rotation=30, ha="right",
                           fontsize=args.tick_fontsize)
        ax.set_xlabel("Bin", fontsize=args.label_fontsize)


# --------------------------------------------------------------------------
# Saving
# --------------------------------------------------------------------------
def save_outputs(stem, bv, n_cv, E_sys, E_stat, E_tot, E_dials, univ, args,
                 univ_sig=None):
    meta_blocks = np.array([f"{b['cont']}:{b['sample']}:{b['var']}"
                            for b in bv.blocks])
    np.savez(stem + ".npz",
             cv=n_cv, cov=E_tot, cov_syst=E_sys, cov_stat=E_stat,
             frac_cov=fractional(E_tot, n_cv), corr=correlation(E_tot),
             blocks=meta_blocks,
             block_offsets=np.array([b["offset"] for b in bv.blocks]),
             **{f"edges__{b['cont']}__{b['sample']}__{b['var']}": b["edges"]
                for b in bv.blocks},
             dials=np.array(list(E_dials)),
             **{f"cov_dial__{d}": E for d, E in E_dials.items()},
             **{f"universes__{d}": np.asarray(u) for d, u in univ.items()},
             **{f"universe_sigmas__{d}": np.asarray(sg)
                for d, sg in (univ_sig or {}).items()})
    if args.no_root:
        return
    try:
        import uproot
    except ImportError:
        return
    idx = np.arange(bv.n + 1, dtype=float)
    with uproot.recreate(stem + ".root") as f:
        f["cv"] = (n_cv, idx)
        f["cov"] = (E_tot, idx, idx)
        f["cov_syst"] = (E_sys, idx, idx)
        f["frac_cov"] = (fractional(E_tot, n_cv), idx, idx)
        f["corr"] = (correlation(E_tot), idx, idx)
        for d, E in E_dials.items():
            f[f"dials/cov_{d}"] = (E, idx, idx)
        for b in bv.blocks:
            f[f"blocks/cv_{b['cont']}_{b['sample']}_{b['var']}"] = (
                n_cv[b["offset"]:b["offset"] + b["nb"]], b["edges"])
