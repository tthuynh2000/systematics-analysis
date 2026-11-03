#!/usr/bin/env python3
"""
syst_covariance.py

Covariance matrices of binned event rates from cross-section dial "universes".

For each varied dial d, the universes are the dial shifted to cv +/- s for
every s in --sigmas (default s = 1, i.e. the -1 and +1 sigma universes).
Every other dial stays at its CV. With N the vector of binned event counts:

    E^(d)_ij = 1/N_univ(d) * sum_{alpha in universes(d)}
               (N_i^alpha - N_i^CV) (N_j^alpha - N_j^CV)

Dials are treated as independent, so the total systematic covariance is

    E_ij = sum_d E^(d)_ij

(averaging over *all* universes of *all* dials instead would divide each
dial's contribution by the number of dials, which is not what you want).
Optionally the MC statistical covariance (diag sum w^2) or Poisson (diag N)
is added.

Outputs (per matrix):
    covariance, fractional covariance  E_ij / (N_i N_j),
    correlation  E_ij / sqrt(E_ii E_jj),
    fractional uncertainty per bin with a per-dial breakdown,
    and an .npz + .root file with everything (incl. each dial's E^(d)).

Binning / samples / selection / weights are identical to
plot_syst_spectra.py, which must sit in the same folder.

Examples
--------
# one matrix per (sample, variable), DUNE atm binning, +/-1 sigma universes
python syst_covariance.py --caf caf.root --weights syst.root \
    --group GENIE_RES --binning atm --outdir cov_genie

# one joint matrix over numu+nue samples and reco Elep, cos(theta_z),
# so cross-sample and cross-variable correlations are included
python syst_covariance.py --caf caf.root --weights syst.root \
    --group FSI_Pion --binning atm --samples numu nue \
    --variables reco_Elep reco_cosz --joint --add-stat mc
"""

import argparse
import os
import sys

import numpy as np
import matplotlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from plot_syst_spectra import (  # noqa: E402
        GROUPS, VARIABLES, SAMPLES, SAMPLE_LABEL, BINNING_PRESETS,
        sample_mask,
        load_caf, derive, read_header, WeightReader, parse_bins, CVWeight,
        apply_detsys_args, detsys_reco_mask,
        CONTAINMENTS, CONT_LABEL, containment_mask, check_containment_args,
        explicit_bin_vars, sample_bins,
        INTERACTIONS, INTERACTION_LABEL, interaction_mask,
        resolve_interaction,
        CVN_NUMU_CUT, CVN_NUE_CUT,
        match_dials, short)
except ImportError as e:
    sys.exit(f"syst_covariance.py needs plot_syst_spectra.py (and "
             f"plot_cv_weights.py) in the same folder: {e}")

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


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
# Default input files (override with --caf / --weights)
DEFAULT_CAF = "/exp/dune/data/users/pgranger/caf_new_sum.2.6M_weighted.root"
DEFAULT_WEIGHTS = "/exp/dune/data/users/tthuynh/merged_new_systs_wsbn_hotfixed_full_header.root"

def parse_args():
    p = argparse.ArgumentParser(
        description="Universe covariance matrices (CV vs cv +/- n sigma) of "
                    "binned event rates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False)
    p.add_argument("--caf", default=DEFAULT_CAF,
                   help="CAF file (cafTree + weights)")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS,
                   help="Weight file (SystWeights + systsHeader)")
    p.add_argument("--max-events", type=int, default=None)

    g = p.add_argument_group("dials / universes")
    g.add_argument("--group", choices=sorted(GROUPS) if GROUPS else None)
    g.add_argument("--dials", nargs="*", default=[])
    g.add_argument("--regex", nargs="*", default=[])
    g.add_argument("--exclude", nargs="*", default=[])
    g.add_argument("--sigmas", nargs="*", type=float, default=[1.0],
                   help="Universes per dial: cv - s and cv + s for each s. "
                        "Default 1 -> the -1 and +1 sigma universes")
    g.add_argument("--cv-dials", nargs="*", default=["*"],
                   help="Dials held at CV in every universe (globs)")
    g.add_argument("--no-cv", action="store_true",
                   help="Do not apply the other dials' CV weights")
    g.add_argument("--add-stat", choices=["none", "mc", "poisson"],
                   default="none",
                   help="Add a diagonal statistical term: MC (sum w^2 of CV) "
                        "or Poisson (N_CV)")

    g = p.add_argument_group("samples, variables, binning")
    g.add_argument("--samples", nargs="*", default=["nue", "numu", "nc"],
                   choices=list(SAMPLES),
                   help="Reco CVN samples. nue+numu adds the nue and numu "
                        "samples into one spectrum; all also adds NC")
    g.add_argument("--variables", nargs="*",
                   default=["reco_Elep", "reco_Ehad", "reco_cosz"],
                   choices=list(VARIABLES))
    g.add_argument("--joint", action="store_true",
                   help="One matrix over all (sample, variable) blocks "
                        "instead of one matrix per block")
    g.add_argument("--binning", choices=sorted(BINNING_PRESETS) + ["none"],
                   default="atm_fcpc",
                   help="atm_fcpc (default): atm + FC/PC nue/numu E_nu, y, "
                        "cosZ edges; atm; none = uniform")
    g.add_argument("--containment", nargs="+", choices=CONTAINMENTS,
                   default=["all"],
                   help="fc / pc / all; several give one matrix each (and "
                        "with --joint one matrix over all of them)")
    g.add_argument("--walldist-cut", type=float, default=20.0,
                   help="FC/PC boundary [cm]")
    g.add_argument("--part-info", default="part_info.npz",
                   help="Output of make_part_info.py; used if it exists")
    g.add_argument("--edges", nargs="*", default=[])
    g.add_argument("--bins", nargs="*", default=[])
    g.add_argument("--nbins", type=int, default=None)
    g.add_argument("--include-nubar", action="store_true")
    g.add_argument("--interaction", choices=INTERACTIONS, default="all",
                   help="True interaction type (mc.nu.iscc): cc, nc, or "
                        "all = CC + NC")
    g.add_argument("--cc-only", action="store_true",
                   help="Shortcut for --interaction cc")
    g.add_argument("--no-quality-cut", action="store_true")
    g.add_argument("--base-weight", choices=["flux_xsec", "final"],
                   default="flux_xsec",
                   help="flux_xsec: unoscillated xsec x true-flavour flux; "
                        "final: final_oscillated_w")
    g.add_argument("--clip-responses", action="store_true",
                   help="Old dial-weight handling: negative -> 0, "
                        "nonfinite -> 1")
    g.add_argument("--independent-b-scans", action="store_true",
                   help="Multiply the CV response of every <provider>_bN "
                        "scan (default: once per provider)")
    g.add_argument("--nparticles-from", choices=["auto", "type", "npandora",
                                                  "npfps"], default="auto",
                   help="nParticles for --cosz mach3 (see "
                        "plot_syst_spectra.py)")
    g.add_argument("--cosz", choices=["mach3", "calo", "systs", "detsys"],
                   default="mach3",
                   help="reco cos(theta_z): mach3 (default, see "
                        "plot_syst_spectra.py), calo, systs or detsys")
    g.add_argument("--selection", choices=["cuts", "argmax", "mach3"],
                   default="cuts",
                   help="Sample selection: CVN thresholds, larger score wins "
                        "(cuts); numu wins (mach3); or the systs.py argmax")
    g.add_argument("--detsys-compatible", action="store_true",
                   help="Match Claire's detsys plots (see "
                        "plot_syst_spectra.py); the band plot is divided by "
                        "bin width, the matrices stay in events")
    g.add_argument("--cvn-numu-cut", type=float, default=CVN_NUMU_CUT,
                   help="numu-like if cvn_numu > this")
    g.add_argument("--cvn-nue-cut", type=float, default=CVN_NUE_CUT,
                   help="nue-like if cvn_nue > this")

    g = p.add_argument_group("plot style")
    g.add_argument("--cmap", default="RdBu_r")
    g.add_argument("--annotate", action="store_true",
                   help="Write values in each cell (small matrices only)")
    g.add_argument("--annotate-max", type=int, default=15)
    g.add_argument("--logx", action="store_true",
                   help="Physical log-scale x axis (and both matrix axes) so "
                        "bins keep their true relative widths. Applies to "
                        "single-block energy plots; cos(theta_z) and --joint "
                        "plots keep the bin-index axis")
    g.add_argument("--logx-zero", type=float, default=None,
                   help="Where to draw a lower edge of 0 on the log axis "
                        "(default: one log-width below the next edge)")
    g.add_argument("--band-universes", type=int, default=5,
                   help="On the total band plot, also draw the +/- sigma "
                        "universe spectra of this many largest dials "
                        "(0 = none). Per-dial band plots always show theirs")
    g.add_argument("--per-dial-plots", action="store_true",
                   help="Also plot each dial's own covariance, correlation "
                        "and error band (in <outdir>/dials/)")
    g.add_argument("--top-dials", type=int, default=8,
                   help="Dials shown individually in the breakdown plot")
    g.add_argument("--short-labels", action="store_true")
    g.add_argument("--per-width", action="store_true",
                   help="Band plot: divide by bin width (matrices unchanged)")
    g.add_argument("--ylabel", default=None, help="Band plot y label")
    g.add_argument("--logy", action="store_true",
                   help="Log y on the CV+band plot")
    g.add_argument("--title", default=None,
                   help="Title prefix (default: group or 'Selected dials')")
    g.add_argument("--title-fontsize", type=float, default=13)
    g.add_argument("--label-fontsize", type=float, default=12)
    g.add_argument("--tick-fontsize", type=float, default=9)
    g.add_argument("--legend-fontsize", type=float, default=8)
    g.add_argument("--legend-ncol", type=int, default=1)
    g.add_argument("--figsize", nargs=2, type=float, default=(8, 7))

    g = p.add_argument_group("output")
    g.add_argument("--outdir", default="syst_cov")
    g.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    g.add_argument("--dpi", type=int, default=150)
    g.add_argument("--no-root", action="store_true",
                   help="Only write .npz, not .root")
    return p.parse_args()


def main():
    args = parse_args()
    matplotlib.use("Agg")

    header = read_header(args.weights)
    patterns = list(args.dials) + (GROUPS.get(args.group, [])
                                   if args.group else [])
    if not patterns and not args.regex:
        sys.exit("Choose dials with --group, --dials or --regex.")
    vary = match_dials(header, patterns, args.regex, args.exclude)
    if not vary:
        sys.exit("No dials matched.")
    cv_dials = [] if args.no_cv else match_dials(header, args.cv_dials)
    sigmas = sorted({abs(s) for s in args.sigmas if s != 0})
    if any(s != 1 for s in sigmas):
        print("  [note] universes beyond +/-1 sigma are averaged in with "
              "1/N_univ: the result is no longer a 1-sigma covariance.")

    print("Reading CAF ...")
    d = load_caf(args.caf, args.max_events, args.part_info)
    check_containment_args(args, d)
    n = len(d["E"])
    resolve_interaction(args)
    if args.detsys_compatible:
        apply_detsys_args(args)
    v = derive(d, args.cvn_numu_cut, args.cvn_nue_cut, args.selection,
               args.cosz, args.base_weight, args.nparticles_from)
    if args.detsys_compatible:
        print(f"  {detsys_reco_mask(v, d)} events without exactly one "
              f"pandora interaction: no reco value")
    for s_name in args.samples:
        print(f"  reco {s_name:8s} sample: "
              f"{np.sum(sample_mask(s_name, v['reco_pdg']))} "
              f"events (before other cuts)")
    wr = WeightReader(args.weights, header, n, raw=not args.clip_responses)

    keep = np.isfinite(v["true_E"]) & (v["true_E"] > 0)
    if not args.include_nubar:
        keep &= d["pdg"] > 0
    keep &= interaction_mask(args.interaction, d["iscc"])
    if not args.no_quality_cut:
        keep &= v["quality"]
    print(f"Events: {n} total, {keep.sum()} after selection "
          f"(true interaction: {args.interaction})")

    preset = None if args.binning == "none" else args.binning
    bins, _ = parse_bins(args.bins, args.nbins, args.edges, preset)
    explicit = explicit_bin_vars(args.bins, args.edges)
    block_list = []
    for c in args.containment:
        cm = (containment_mask(c, d["walldist_min"], args.walldist_cut)
              if c != "all" else np.ones(n, bool))
        for s in args.samples:
            for var in args.variables:
                e, special = sample_bins(var, s, c, bins, preset, explicit)
                if special:
                    print(f"  binning {c}/{s}/{var}: {e.tolist()}")
                block_list.append((s, var, e, c, cm))
    groups = ([block_list] if args.joint else [[b] for b in block_list])
    bvs = [BinVector(g, v, keep) for g in groups]

    # ---- CV weight ----------------------------------------------------------
    print(f"Building CV weight from {len(cv_dials)} dial(s) ...")
    base = v["base_w"]
    cvw = CVWeight(wr, header, cv_dials, keep_cached=vary,
                   independent_b=args.independent_b_scans)
    others_at_cv = cvw.others
    w_cv = base * cvw.total
    n_cv = [bv(w_cv) for bv in bvs]
    stat_mc = [bv(w_cv ** 2) for bv in bvs]

    # ---- universes per dial -------------------------------------------------
    E_dials = [dict() for _ in bvs]
    univ = [dict() for _ in bvs]
    univ_sig = [dict() for _ in bvs]
    for dial in vary:
        info = header[dial]
        if dial not in wr.t.keys():
            print(f"  [skip] {dial}: not in SystWeights")
            continue
        if len(info["variations"]) <= 1:
            print(f"  [skip] {dial}: correction / single-valued dial, "
                  f"no universes")
            wr.cache.pop(dial, None)
            continue
        rest = base * others_at_cv(dial)
        vals, labs = [], []
        for s in sigmas:
            for sg in (-1, +1):
                val = info["cv"] + sg * s
                w = wr.at_value(dial, val)
                if w is None:
                    print(f"  [info] {short(dial, args.short_labels)}: "
                          f"{sg * s:+g} sigma not available")
                    continue
                vals.append(rest * w)
                labs.append(sg * s)
        if not vals:
            print(f"  [skip] {dial}: no universes (correction / single-"
                  f"valued dial)")
            wr.cache.pop(dial, None)
            continue
        for k, bv in enumerate(bvs):
            nu = [bv(w) for w in vals]
            univ[k][dial] = nu
            univ_sig[k][dial] = labs
            E_dials[k][dial] = cov_from_universes(n_cv[k], nu)
        print(f"  {dial}: {len(vals)} universe(s)")
        wr.cache.pop(dial, None)

    if not E_dials[0]:
        sys.exit("No dial produced universes; nothing to do.")

    # ---- combine, plot, save -----------------------------------------------
    os.makedirs(args.outdir, exist_ok=True)
    if args.logx:
        for bv in bvs:
            if len(bv.blocks) > 1:
                print("  [info] --logx ignored for the joint matrix "
                      "(blocks are concatenated on a bin-index axis)")
            elif bv.blocks[0]["edges"][0] < 0:
                print(f"  [info] --logx ignored for "
                      f"{bv.blocks[0]['sample']}:{bv.blocks[0]['var']} "
                      f"(negative edges)")
            elif bv.blocks[0]["edges"][0] == 0:
                c, _, _ = axis_coords(bv, args)
                print(f"  [info] {bv.blocks[0]['sample']}:"
                      f"{bv.blocks[0]['var']}: lower edge 0 drawn at "
                      f"{c[0]:.3g} on the log axis (labelled 0)")
    prefix = args.title or (args.group or "Selected dials")
    for k, bv in enumerate(bvs):
        E_sys = sum(E_dials[k].values())
        if args.add_stat == "mc":
            E_stat = np.diag(stat_mc[k])
        elif args.add_stat == "poisson":
            E_stat = np.diag(n_cv[k])
        else:
            E_stat = np.zeros_like(E_sys)
        E_tot = E_sys + E_stat

        if args.joint:
            tag = "joint"
            what = " + ".join(f"{b['cont']}:{b['sample']}:{b['var']}"
                              for b in bv.blocks)
            sub = "joint: " + ", ".join(
                f"{CONT_LABEL[b['cont']]}{SHORT_SAMPLE[b['sample']]} "
                f"{SHORT_VAR[b['var']]}"
                for b in bv.blocks) + INTERACTION_LABEL[args.interaction]
        else:
            b = bv.blocks[0]
            cpre = "" if b["cont"] == "all" else f"{b['cont']}_"
            tag = f"{cpre}{b['sample']}_{b['var']}"
            what = f"{cpre}{b['sample']}:{b['var']}"
            sub = (f"{CONT_LABEL[b['cont']]}{SAMPLE_LABEL[b['sample']]}"
                   f"{INTERACTION_LABEL[args.interaction]}, "
                   f"{SHORT_VAR[b['var']]}")
        stat_note = {"none": "", "mc": " + MC stat",
                     "poisson": " + Poisson stat"}[args.add_stat]

        miss = missing_bins(E_tot) | ~np.isfinite(n_cv[k])
        good = ~miss
        Eg = E_tot[np.ix_(good, good)]
        eig = np.linalg.eigvalsh(Eg) if good.any() else np.array([0.0])
        sym = np.max(np.abs(Eg - Eg.T)) if good.any() else 0.0
        print(f"[{what}] {bv.n} bins ({miss.sum()} missing: nonfinite "
              f"response), {len(E_dials[k])} dial(s); "
              f"total CV = {np.nansum(n_cv[k]):.6g}; "
              f"min eigenvalue = {eig.min():.3g} "
              f"(rank {np.sum(eig > 1e-12 * max(eig.max(), 1e-300))}); "
              f"max |E - E^T| = {sym:.2g}")

        stem = os.path.join(args.outdir, tag)
        fmt = args.format
        plot_matrix(E_tot, bv, f"{prefix}: covariance{stat_note}\n{sub}",
                    r"$E_{ij}$ [events$^2$]", f"{stem}_cov.{fmt}", args,
                    "cov")
        plot_matrix(fractional(E_tot, n_cv[k]), bv,
                    f"{prefix}: fractional covariance{stat_note}\n{sub}",
                    r"$E_{ij}/(N_i N_j)$", f"{stem}_fraccov.{fmt}", args,
                    "frac")
        plot_matrix(correlation(E_tot), bv,
                    f"{prefix}: correlation{stat_note}\n{sub}",
                    r"$\rho_{ij}$", f"{stem}_corr.{fmt}", args, "corr")
        plot_breakdown(n_cv[k], E_tot, E_dials[k], bv,
                       f"{prefix}: fractional uncertainty{stat_note}\n{sub}",
                       f"{stem}_fracerr.{fmt}", args)
        tot_curves = []
        if args.band_universes:
            import matplotlib.pyplot as plt
            ranked = sorted(E_dials[k], key=lambda dd: -np.trace(E_dials[k][dd]))
            shown = ranked[:args.band_universes]
            cmap = plt.get_cmap("tab10")
            for j, dd in enumerate(shown):
                tot_curves += universe_curves(
                    univ[k][dd], univ_sig[k][dd],
                    label_prefix=short(dd, args.short_labels) + " ",
                    color=cmap((j + 1) % 10), lw=1.2)   # skip band blue
            if len(ranked) > len(shown):
                print(f"  [info] band plot shows universes of the "
                      f"{len(shown)} largest of {len(ranked)} dials "
                      f"(--band-universes)")
        plot_band(n_cv[k], E_tot, bv,
                  f"{prefix}: CV with $\\pm\\sqrt{{E_{{ii}}}}${stat_note}"
                  f"\n{sub}", f"{stem}_band.{fmt}", args, curves=tot_curves)
        save_outputs(stem, bv, n_cv[k], E_sys, E_stat, E_tot, E_dials[k],
                     univ[k], args, univ_sig[k])

        if args.per_dial_plots:
            ddir = os.path.join(args.outdir, "dials")
            os.makedirs(ddir, exist_ok=True)
            for dial, Ed in E_dials[k].items():
                lab = short(dial, args.short_labels)
                dstem = os.path.join(ddir, f"{tag}__{dial}")
                plot_matrix(Ed, bv, f"{lab}: covariance\n{sub}",
                            r"$E^{(d)}_{ij}$ [events$^2$]",
                            f"{dstem}_cov.{fmt}", args, "cov")
                plot_matrix(correlation(Ed), bv, f"{lab}: correlation\n{sub}",
                            r"$\rho_{ij}$", f"{dstem}_corr.{fmt}", args, "corr")
                plot_band(n_cv[k], Ed, bv,
                          f"{lab}: CV with $\\pm\\sqrt{{E^{{(d)}}_{{ii}}}}$"
                          f"\n{sub}", f"{dstem}_band.{fmt}", args,
                          band_label=r"$\pm\sqrt{E^{(d)}_{ii}}$ (" + lab + ")",
                          curves=universe_curves(univ[k][dial],
                                                 univ_sig[k][dial]))

    print(f"Saved matrices, plots and .npz/.root files under {args.outdir}/")


if __name__ == "__main__":
    main()
