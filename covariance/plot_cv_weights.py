#!/usr/bin/env python3
"""
plot_cv_weights.py

Plot per-event systematic weights at a chosen sigma (default: sigma = 0, the
central value) for a group of dials stored in a weight CAF/ROOT file.

How it works
------------
1. systsHeader holds one entry per dial: `name`, `cv`, `variations`, `id`,
   `isCorrection`. For each selected dial, the script finds the index of the
   dial's own `cv` value (or --sigma, if given) inside `variations`.
       e.g. variations = [0, 1, -1, 2, -2, 3, -3]  ->  sigma 0 is index 0
2. In SystWeights, each dial branch stores, per event, an array of weights in
   the SAME order as `variations`. The script takes element [index] for every
   event.
3. It histograms those weights (evenly spaced bins, count of events on Y).

Requirements: uproot, awkward, numpy, matplotlib
    pip install uproot awkward numpy matplotlib

Examples
--------
# List every dial, its variations, and where sigma=0 sits
python plot_cv_weights.py weights.root --list

# CV weights for all CCQE template dials, 50 bins, overlaid
python plot_cv_weights.py weights.root --group CCQETemplate_HFToCRPA --nbins 50

# Explicit dials / glob patterns, separate panel per dial, log y
python plot_cv_weights.py weights.root \
    --dials "GENIEReWeight_ICARUS_v2_multisigma_Fr*" --layout grid --logy

# The +1 sigma weights instead of the CV
python plot_cv_weights.py weights.root --group MEC_XSecShape --sigma 1
"""

import argparse
import fnmatch
import re
import sys

import numpy as np

try:
    import uproot
    import awkward as ak
except ImportError:
    sys.exit("This script needs uproot and awkward: pip install uproot awkward")

import matplotlib

# --------------------------------------------------------------------------
# Predefined dial groups (glob patterns). Edit / extend freely.
# --------------------------------------------------------------------------
GROUPS = {
    # CCQE axial form factor: same physics, different parameterizations
    "CCQE_Axial_GENIE_ZExp": [
        "GENIEReWeight_ICARUS_v2_multisigma_ZExpA[1-4]CCQE",
    ],
    "CCQE_Axial_Deut_PCA": [
        "ZExpPCAWeighter_SBN_v3_Deut_b[1-4]",
    ],
    "CCQE_Axial_MvA_PCA": [
        "ZExpPCAWeighter_SBN_v3_MvA_b[1-4]",
    ],

    # CCQE vector form factors
    "CCQE_Vector": [
        "GENIEReWeight_ICARUS_v2_multisigma_VecFFCCQEshape",
    ],

    # CCQE nuclear-model templates
    "CCQETemplate_HFToCRPA": [
        "CCQETemplateReweight_SBN_v3_HFToCRPA_q0bin[0-4]",
    ],
    "CCQETemplate_LFGToHF": [
        "CCQETemplateReweight_SBN_v3_LFGToHF_q0bin[0-4]",
    ],
    "CCQETemplate_LFGToSF": [
        "CCQETemplateReweight_SBN_v3_LFGToSF_q0bin[0-4]",
    ],

    # QE interference and central-model correction
    "QEInterference": [
        "QEInterference_SBN_v3_QEIntf_dial_[0-5]",
    ],
    "CCQEXSecCorr": [
        "CCQEXSecCorr_SBN_v3_CCQEXSecCorr",
    ],

    # NC elastic scattering
    "GENIE_NCEL": [
        "GENIEReWeight_ICARUS_v2_multisigma_MaNCEL",
        "GENIEReWeight_ICARUS_v2_multisigma_EtaNCEL",
    ],

    # Standard resonance production (CC + NC) and shared decay dials
    "GENIE_RES": [
        "GENIEReWeight_ICARUS_v2_multisigma_M[av]CCRES",
        "GENIEReWeight_ICARUS_v2_multisigma_M[av]NCRES",
        "GENIEReWeight_ICARUS_v2_multisigma_RDecBR1eta",
        "GENIEReWeight_ICARUS_v2_multisigma_RDecBR1gamma",
        "GENIEReWeight_ICARUS_v2_multisigma_Theta_Delta2Npi",
    ],

    # ResIso production and the same shared decay dials
    "ResIso_RES": [
        "ResIso_ResIso_ResonanceIsolation_MaRes_ResCode[0-1]",
        "ResIso_ResIso_ResonanceIsolation_MvRes_ResCode[0-1]",
        "GENIEReWeight_ICARUS_v2_multisigma_RDecBR1eta",
        "GENIEReWeight_ICARUS_v2_multisigma_RDecBR1gamma",
        "GENIEReWeight_ICARUS_v2_multisigma_Theta_Delta2Npi",
    ],

    # Nonresonant inelastic backgrounds
    "GENIE_NonRES_CC": [
        "GENIEReWeight_ICARUS_v2_multisigma_NonRESBG*CC[1-2]pi",
    ],
    "GENIE_NonRES_NC": [
        "GENIEReWeight_ICARUS_v2_multisigma_NonRESBG*NC[1-2]pi",
    ],

    # DIS / Bodek-Yang
    "GENIE_DIS_BY": [
        "GENIEReWeight_ICARUS_v2_multisigma_[AB]htBY",
        "GENIEReWeight_ICARUS_v2_multisigma_CV[1-2]uBY",
    ],

    # MEC / 2p2h parameter-level variations
    "MEC_EnergyDependence": [
        "GENIEReWeight_ICARUS_v2_multisigma_EnergyDependence_CCMEC",
    ],
    "MEC_ComponentFractions": [
        "GENIEReWeight_ICARUS_v2_multisigma_FracDelta_CCMEC",
        "GENIEReWeight_ICARUS_v2_multisigma_FracPN_CCMEC",
    ],
    "MEC_XSecShape": [
        "GENIEReWeight_ICARUS_v2_multisigma_XSecShape_CCMEC",
        "GENIEReWeight_ICARUS_v2_multisigma_XSecShape_CCMEC_Empirical",
        "GENIEReWeight_ICARUS_v2_multisigma_XSecShape_CCMEC_Martini",
    ],
    "MEC_DecayKinematics": [
        "GENIEReWeight_ICARUS_v2_multisigma_DecayAngMEC",
        "GENIEReWeight_ICARUS_v2_multisigma_DecayAng2MEC",
        "GENIEReWeight_ICARUS_v2_multisigma_DecayAngMECVariationResponse",
    ],

    # MEC model templates
    "MECTemplate_SuSAToMartini": [
        "MECq0q3InterpWeighting_SBN_v3_SuSAToMar_MECResponse_q0bin[0-3]",
    ],
    "MECTemplate_SuSAToValencia": [
        "MECq0q3InterpWeighting_SBN_v3_SuSAToVal_MECResponse_q0bin[0-3]",
    ],

    # MEC nuclear-target scaling
    "MEC_C12ToAr40_Scaling": [
        "MiscInteractionSysts_DUNE_TDR_FakeData_C12ToAr40_2p2hScaling_nu",
        "MiscInteractionSysts_DUNE_TDR_FakeData_C12ToAr40_2p2hScaling_nubar",
    ],

    # Final-state interactions
    "FSI_Nucleon": [
        "GENIEReWeight_ICARUS_v2_multisigma_FrAbs_N",
        "GENIEReWeight_ICARUS_v2_multisigma_FrCEx_N",
        "GENIEReWeight_ICARUS_v2_multisigma_FrInel_N",
        "GENIEReWeight_ICARUS_v2_multisigma_FrPiProd_N",
        "GENIEReWeight_ICARUS_v2_multisigma_MFP_N",
    ],
    "FSI_Pion": [
        "GENIEReWeight_ICARUS_v2_multisigma_FrAbs_pi",
        "GENIEReWeight_ICARUS_v2_multisigma_FrCEx_pi",
        "GENIEReWeight_ICARUS_v2_multisigma_FrInel_pi",
        "GENIEReWeight_ICARUS_v2_multisigma_FrPiProd_pi",
        "GENIEReWeight_ICARUS_v2_multisigma_MFP_pi",
    ],
    "PionAbs_QuasiDeuteron": [
        "PionAbsWeighter_SBN_v3_QuasiDeuteronFraction",
    ],

    # Nuclear density / geometry
    "WoodsSaxon": [
        "WSReweight_v1_nucleus_radius",
        "WSReweight_v1_surface_thickness",
    ],

    # Flavor and neutrino/antineutrino cross-section ratios
    "Flavor_NuNuBar_Ratios": [
        "MiscInteractionSysts_DUNE_TDR_FakeData_nuenuebar_xsec_ratio",
        "MiscInteractionSysts_DUNE_TDR_FakeData_nuenumu_xsec_ratio",
    ],
}


# --------------------------------------------------------------------------
# Header handling
# --------------------------------------------------------------------------
def read_header(path, tree_name):
    """Return {dial_name: dict(cv, variations, id, isCorrection)}."""
    with uproot.open(path) as f:
        t = f[tree_name]
        wanted = [b for b in ("name", "cv", "variations", "id", "isCorrection")
                  if b in t.keys()]
        arr = t.arrays(wanted, library="ak")
    cols = {b: ak.to_list(arr[b]) for b in wanted}
    header = {}
    for i, name in enumerate(cols["name"]):
        name = name.decode() if isinstance(name, bytes) else str(name)
        header[name] = {
            "cv": cols.get("cv", [None] * len(cols["name"]))[i],
            "variations": [float(v) for v in (cols["variations"][i] or [])],
            "id": cols.get("id", [i] * len(cols["name"]))[i],
            "isCorrection": bool(cols.get("isCorrection",
                                          [False] * len(cols["name"]))[i]),
        }
    return header


def target_value(info, sigma):
    """Variation value to look up: --sigma if given, else systsHeader/cv."""
    if sigma is not None:
        return float(sigma)
    cv = info.get("cv")
    return float(cv) if cv is not None else 0.0


def find_sigma_index(info, sigma, tol=1e-6):
    """Index of the target value (header cv by default) in `variations`.

    Returns (index, target, note). index is None if the dial cannot be used.
    """
    target = target_value(info, sigma)
    vs = info["variations"]
    for i, v in enumerate(vs):
        if abs(v - target) < tol:
            return i, target, ""
    # Corrections / single-valued dials store one weight per event
    if len(vs) <= 1:
        return 0, target, "single-valued/correction dial: using element 0"
    return None, target, f"value {target:g} not in variations {vs}"


def select_dials(all_names, patterns, regexes, excludes):
    chosen = []
    for name in all_names:
        hit = any(fnmatch.fnmatchcase(name, p) for p in patterns) or \
              any(re.search(r, name) for r in regexes)
        if hit and not any(fnmatch.fnmatchcase(name, e) for e in excludes):
            chosen.append(name)
    return chosen


# --------------------------------------------------------------------------
# Weight extraction
# --------------------------------------------------------------------------
def extract_weights(paths, tree_name, dial, index, missing):
    """Per-event weight at `index` for `dial`, concatenated over all files."""
    pieces = []
    for p in paths:
        with uproot.open(p) as f:
            t = f[tree_name]
            if dial not in t.keys():
                print(f"  [warn] {dial} not in {tree_name} of {p}, skipping file")
                continue
            w = t[dial].array(library="ak")
        if w.ndim == 1:                      # scalar branch (one weight/event)
            pieces.append(ak.to_numpy(w).astype(float))
            continue
        n = ak.num(w, axis=1)
        out = np.full(len(w), np.nan)
        ok = ak.to_numpy(n > index)
        if ok.any():
            out[ok] = ak.to_numpy(w[ok][:, index]).astype(float)
        if (~ok).any():
            print(f"  [info] {dial}: {(~ok).sum()} events with fewer than "
                  f"{index + 1} entries -> {missing}")
        pieces.append(out)
    if not pieces:
        return np.array([])
    vals = np.concatenate(pieces)
    if missing == "one":
        vals = np.where(np.isnan(vals), 1.0, vals)
    bad = ~np.isfinite(vals)
    if bad.any():
        print(f"  [info] {dial}: dropping {bad.sum()} NaN/inf weights")
    return vals[~bad]


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def make_bins(arrays, nbins, xrange, logx=False):
    if xrange is not None:
        lo, hi = xrange
    else:
        nonempty = [a for a in arrays if a.size]
        lo = min(a.min() for a in nonempty)
        hi = max(a.max() for a in nonempty)
    if logx:
        if lo <= 0:
            sys.exit("--logx needs a positive x range (check --range).")
        if np.isclose(lo, hi):
            lo, hi = lo / 1.05, hi * 1.05
        return np.geomspace(lo, hi, nbins + 1)   # even in log(x)
    if np.isclose(lo, hi):                    # e.g. every CV weight == 1
        pad = 0.05 * max(abs(lo), 1.0)
        lo, hi = lo - pad, hi + pad
    return np.linspace(lo, hi, nbins + 1)


def short_label(name, strip_prefix):
    if not strip_prefix:
        return name
    for pre in ("GENIEReWeight_ICARUS_v2_multisigma_",
                "CCQETemplateReweight_SBN_v3_",
                "MECq0q3InterpWeighting_SBN_v3_",
                "MiscInteractionSysts_DUNE_TDR_",
                "ZExpPCAWeighter_SBN_v3_",
                "QEInterference_SBN_v3_",
                "PionAbsWeighter_SBN_v3_",
                "CCQEXSecCorr_SBN_v3_",
                "ResIso_ResIso_",
                "WSReweight_v1_"):
        if name.startswith(pre):
            return name[len(pre):]
    return name


def stats_text(w):
    return (f"N={w.size}, mean={w.mean():.4g}, RMS={w.std():.3g}, "
            f"min={w.min():.3g}, max={w.max():.3g}")


def plot(data, args):
    import matplotlib.pyplot as plt

    names = list(data)
    hist_kw = dict(histtype=args.histtype, linewidth=args.linewidth,
                   alpha=args.alpha, density=args.density)
    ylabel = args.ylabel or ("Normalized events" if args.density else "Events")
    point = ("central value (header cv)" if args.sigma is None
             else f"$\\sigma$ = {args.sigma:g}")
    xlabel = args.xlabel or f"Weight at {point}"
    title = args.title or f"{args.group or 'Selected dials'}: {point} weights"

    def label_for(n):
        lab = short_label(n, args.short_labels)
        if args.legend_stats:
            lab += "\n" + stats_text(data[n])
        return lab

    if args.layout == "overlay":
        fig, ax = plt.subplots(figsize=args.figsize)
        bins = make_bins(list(data.values()), args.nbins, args.range,
                         args.logx)
        cmap = plt.get_cmap(args.cmap)
        ncolors = getattr(cmap, "N", 256)
        for i, n in enumerate(names):
            if ncolors <= 20:            # qualitative map: take colours in order
                color = cmap(i % ncolors)
            else:                        # continuous map: spread across it
                color = cmap(i / max(len(names) - 1, 1))
            ax.hist(data[n], bins=bins, label=label_for(n), color=color,
                    **hist_kw)
        axes = [ax]
        ax.set_title(title, fontsize=args.title_fontsize)
        ax.set_xlabel(xlabel, fontsize=args.label_fontsize)
        ax.set_ylabel(ylabel, fontsize=args.label_fontsize)
    else:  # grid
        ncol = args.ncols
        nrow = int(np.ceil(len(names) / ncol))
        figsize = args.figsize or (4.5 * ncol, 3.5 * nrow)
        fig, axs = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False,
                                sharex=args.sharex)
        axes = axs.ravel()
        common = (make_bins(list(data.values()), args.nbins, args.range,
                            args.logx) if args.sharex else None)
        for ax, n in zip(axes, names):
            bins = common if common is not None else \
                make_bins([data[n]], args.nbins, args.range, args.logx)
            ax.hist(data[n], bins=bins, label=label_for(n), **hist_kw)
            ax.set_xlabel(xlabel, fontsize=args.label_fontsize * 0.8)
            ax.set_ylabel(ylabel, fontsize=args.label_fontsize * 0.8)
        for ax in axes[len(names):]:
            ax.set_visible(False)
        fig.suptitle(title, fontsize=args.title_fontsize)
        axes = axes[:len(names)]

    for ax in axes:
        if args.logx:
            ax.set_xscale("log")
        if args.logy:
            ax.set_yscale("log")
        if args.grid_lines:
            ax.grid(alpha=0.3)
        if args.legend:
            ax.legend(loc=args.legend_loc, fontsize=args.legend_fontsize,
                      ncol=args.legend_ncol, frameon=True)
        if args.vline is not None:
            ax.axvline(args.vline, color="k", ls="--", lw=1)

    fig.tight_layout()
    if args.output:
        fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved {args.output}")
    if args.show:
        plt.show()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
# Default weight file (override by passing file(s) on the command line)
DEFAULT_WEIGHTS = "/exp/dune/data/users/tthuynh/merged_new_systs_wsbn_hotfixed_full_header.root"

def parse_args():
    p = argparse.ArgumentParser(
        description="Histogram per-event systematic weights at a given sigma "
                    "(default CV, sigma=0) for a group of dials.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False)
    p.add_argument("files", nargs="*", default=[DEFAULT_WEIGHTS],
                   help="Input weight ROOT file(s); default: DEFAULT_WEIGHTS")

    g = p.add_argument_group("trees")
    g.add_argument("--header-tree", default="systsHeader")
    g.add_argument("--weight-tree", default="SystWeights")

    g = p.add_argument_group("dial selection")
    g.add_argument("--list", action="store_true",
                   help="Print all dials with variations and sigma index, then exit")
    g.add_argument("--group", choices=sorted(GROUPS),
                   help="Predefined dial group (see GROUPS in the script)")
    g.add_argument("--dials", nargs="*", default=[],
                   help="Dial names or glob patterns, e.g. 'ZExpPCA*'")
    g.add_argument("--regex", nargs="*", default=[],
                   help="Regular expressions matched against dial names")
    g.add_argument("--exclude", nargs="*", default=[],
                   help="Glob patterns to drop from the selection")
    g.add_argument("--sigma", type=float, default=None,
                   help="Variation value to plot. Default: each dial's cv "
                        "from systsHeader/cv")
    g.add_argument("--missing", choices=["drop", "one"], default="drop",
                   help="Events lacking an entry at the sigma index: drop them "
                        "or treat their weight as 1")
    g.add_argument("--skip-trivial", action="store_true",
                   help="Skip dials whose weights are all exactly 1")

    g = p.add_argument_group("binning")
    g.add_argument("--nbins", type=int, default=50, help="Number of even bins")
    g.add_argument("--range", nargs=2, type=float, metavar=("XMIN", "XMAX"),
                   help="X range (default: min to max of the weights)")

    g = p.add_argument_group("plot style")
    g.add_argument("--layout", choices=["overlay", "grid"], default="overlay")
    g.add_argument("--ncols", type=int, default=3, help="Columns in grid layout")
    g.add_argument("--sharex", action="store_true",
                   help="Grid: same binning in every panel")
    g.add_argument("--title", default=None)
    g.add_argument("--xlabel", default=None)
    g.add_argument("--ylabel", default=None)
    g.add_argument("--title-fontsize", type=float, default=14)
    g.add_argument("--label-fontsize", type=float, default=12)
    g.add_argument("--logx", action="store_true",
                   help="Log x axis with log-spaced bins (weights <= 0 dropped)")
    g.add_argument("--logy", action="store_true", help="Log y axis")
    g.add_argument("--density", action="store_true",
                   help="Normalize each histogram to unit area")
    g.add_argument("--histtype", default="step",
                   choices=["step", "stepfilled", "bar"])
    g.add_argument("--linewidth", type=float, default=1.5)
    g.add_argument("--alpha", type=float, default=0.9)
    g.add_argument("--cmap", default="tab10", help="Matplotlib colormap (overlay)")
    g.add_argument("--no-legend", dest="legend", action="store_false")
    g.add_argument("--legend-loc", default="best")
    g.add_argument("--legend-fontsize", type=float, default=8)
    g.add_argument("--legend-ncol", type=int, default=1)
    g.add_argument("--legend-stats", action="store_true",
                   help="Add N/mean/RMS/min/max to each legend entry")
    g.add_argument("--short-labels", action="store_true",
                   help="Strip the generator prefix from dial names in labels")
    g.add_argument("--grid-lines", action="store_true")
    g.add_argument("--vline", type=float, default=None,
                   help="Draw a dashed vertical line at this x (e.g. 1)")
    g.add_argument("--figsize", nargs=2, type=float, default=None,
                   metavar=("W", "H"))

    g = p.add_argument_group("output")
    g.add_argument("-o", "--output", default="cv_weights.png",
                   help="Output image (png/pdf/svg); '' to skip saving")
    g.add_argument("--dpi", type=int, default=150)
    g.add_argument("--show", action="store_true", help="Open an interactive window")
    args = p.parse_args()
    if args.figsize is None and args.layout == "overlay":
        args.figsize = (9, 6)
    return args


def main():
    args = parse_args()
    if not args.show:
        matplotlib.use("Agg")

    header = read_header(args.files[0], args.header_tree)

    if args.list:
        print(f"{'dial':70s} {'isCorr':>6s} {'target':>7s} {'index':>5s}  variations")
        for name, info in header.items():
            idx, tgt, _ = find_sigma_index(info, args.sigma)
            print(f"{name:70s} {str(info['isCorrection']):>6s} {tgt:>7g} "
                  f"{str(idx):>5s}  {info['variations']}")
        return

    patterns = list(args.dials) + (GROUPS[args.group] if args.group else [])
    if not patterns and not args.regex:
        sys.exit("Choose dials with --group, --dials or --regex (or --list).")
    dials = select_dials(header.keys(), patterns, args.regex, args.exclude)
    if not dials:
        sys.exit("No dials matched the selection.")

    print(f"Selected {len(dials)} dial(s):")
    data = {}
    for d in dials:
        idx, tgt, note = find_sigma_index(header[d], args.sigma)
        if idx is None:
            print(f"  [skip] {d}: {note}")
            continue
        w = extract_weights(args.files, args.weight_tree, d, idx, args.missing)
        if w.size == 0:
            print(f"  [skip] {d}: no weights")
            continue
        if args.skip_trivial and np.all(w == 1.0):
            print(f"  [skip] {d}: all weights are exactly 1")
            continue
        data[d] = w
        print(f"  {d}: value {tgt:g} -> index {idx} "
              f"{('(' + note + ')') if note else ''}\n"
              f"      {stats_text(w)}")

    if args.logx:
        for d in list(data):
            nonpos = data[d] <= 0
            if nonpos.any():
                print(f"  [info] {d}: dropping {nonpos.sum()} weights <= 0 "
                      f"for --logx")
                data[d] = data[d][~nonpos]
            if data[d].size == 0:
                del data[d]

    if not data:
        sys.exit("Nothing to plot.")

    # Flag dials with event-by-event identical weights (they overlap exactly)
    groups, seen = [], set()
    keys = list(data)
    for i, a in enumerate(keys):
        if a in seen:
            continue
        same = [a] + [b for b in keys[i + 1:] if b not in seen
                      and data[a].shape == data[b].shape
                      and np.array_equal(data[a], data[b])]
        if len(same) > 1:
            seen.update(same)
            groups.append(same)
    for grp in groups:
        print(f"  [note] identical weights, histograms overlap exactly "
              f"(try --layout grid): {', '.join(grp)}")
    plot(data, args)


if __name__ == "__main__":
    main()
