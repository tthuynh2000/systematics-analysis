#!/usr/bin/env python3
"""
syst_covariance_josh.py

Same covariance matrices / error bands as syst_covariance.py, but with JOSH's
selection, reconstruction and weighting (see plot_syst_spectra_josh.py for
the full definitions). Needs plot_syst_spectra_josh.py, plot_syst_spectra.py,
syst_covariance.py and plot_cv_weights.py in the same folder.

One-dial convention
-------------------
Every dial is evaluated on its own, on top of the nominal weight
xsec * matching-flavour flux (no other dial's CV is applied):

    ref_d(i)   = histogram of  w = nominal * response_d(cv)
    univ_d(i)  = histogram of  w = nominal * response_d(cv +/- s)
    E^(d)_ij   = 1/N_univ sum_univ (univ_i - ref_i)(univ_j - ref_j)
    E_ij       = sum_d E^(d)_ij
    band       : nominal (no dial) +/- sqrt(E_ii)

response = stored dial weight, nonfinite -> 1, clipped to [-9999, 9999];
weights <= 0 are set to 0. Per-dial band plots are drawn around ref_d.

Example
-------
python syst_covariance_josh.py --group CCQE_Axial_Deut_PCA \\
    --containment fc pc --samples cc_nue cc_numu \\
    --variables reco_E reco_y reco_cosz --per-dial-plots
"""

import argparse
import os
import sys

import numpy as np
import matplotlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import plot_syst_spectra_josh as J
    from plot_syst_spectra_josh import (  # noqa: E402
        GROUPS, DEFAULT_CAF, DEFAULT_WEIGHTS, CONT_LABEL, SAMPLE_LABEL,
        JOSH_SAMPLE_LABEL, WeightReader, match_dials, read_header, parse_bins,
        explicit_bin_vars, short, josh_weight, josh_bins, prepare,
        add_josh_args, add_dial_args, add_binning_args)
    import syst_covariance as C
    from syst_covariance import (  # noqa: E402
        BinVector, SHORT_SAMPLE, SHORT_VAR, cov_from_universes, correlation,
        fractional, missing_bins, plot_matrix, plot_breakdown, plot_band,
        universe_curves, save_outputs, axis_coords)
except ImportError as e:
    sys.exit(f"syst_covariance_josh.py needs plot_syst_spectra_josh.py, "
             f"plot_syst_spectra.py, syst_covariance.py and plot_cv_weights.py "
             f"in the same folder: {e}")

SHORT_SAMPLE.update({
    "cc_nue": r"CC$\nu_e$", "cc_numu": r"CC$\nu_\mu$", "nc": "NC",
    "cc_nue+cc_numu": r"CC$\nu_e$+CC$\nu_\mu$", "all": r"CC$\nu_e$+CC$\nu_\mu$+NC"})
SHORT_VAR.update({"true_Ehad": r"$q_0$", "true_Elep": r"$E_\nu - q_0$"})


def parse_args():
    p = argparse.ArgumentParser(
        description="Covariance matrices from +/- sigma universes, Josh's "
                    "selection, reconstruction and one-dial weights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False)
    p.add_argument("--caf", default=DEFAULT_CAF)
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--max-events", type=int, default=None)
    g = add_dial_args(p)
    g.add_argument("--sigmas", nargs="*", type=float, default=[1.0])
    g.add_argument("--add-stat", choices=["none", "mc", "poisson"],
                   default="none")
    add_josh_args(p)
    g = add_binning_args(p)
    g.add_argument("--joint", action="store_true")

    g = p.add_argument_group("plot style")
    g.add_argument("--cmap", default="RdBu_r")
    g.add_argument("--annotate", action="store_true")
    g.add_argument("--annotate-max", type=int, default=15)
    g.add_argument("--logx", action="store_true")
    g.add_argument("--logx-zero", type=float, default=None)
    g.add_argument("--band-universes", type=int, default=5)
    g.add_argument("--per-dial-plots", action="store_true")
    g.add_argument("--top-dials", type=int, default=8)
    g.add_argument("--short-labels", action="store_true")
    g.add_argument("--per-width", action="store_true")
    g.add_argument("--ylabel", default=None)
    g.add_argument("--logy", action="store_true")
    g.add_argument("--title", default=None)
    g.add_argument("--title-fontsize", type=float, default=13)
    g.add_argument("--label-fontsize", type=float, default=12)
    g.add_argument("--tick-fontsize", type=float, default=9)
    g.add_argument("--legend-fontsize", type=float, default=8)
    g.add_argument("--legend-ncol", type=int, default=1)
    g.add_argument("--figsize", nargs=2, type=float, default=(8, 7))

    g = p.add_argument_group("output")
    g.add_argument("--outdir", default="syst_cov_josh")
    g.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    g.add_argument("--dpi", type=int, default=150)
    g.add_argument("--no-root", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    matplotlib.use("Agg")
    header = read_header(args.weights)
    if args.list:
        for nme, i in header.items():
            print(f"{nme:70s} {i['cv']:>5g}  {i['variations']}")
        return
    patterns = list(args.dials) + (GROUPS.get(args.group, [])
                                   if args.group else [])
    if not patterns and not args.regex:
        sys.exit("Choose dials with --group, --dials or --regex.")
    vary = match_dials(header, patterns, args.regex, args.exclude)
    if not vary:
        sys.exit("No dials matched.")
    sigmas = sorted({abs(s) for s in args.sigmas if s != 0})
    if any(s != 1 for s in sigmas):
        print("  [note] universes beyond +/-1 sigma are averaged in with "
              "1/N_univ: the result is no longer a 1-sigma covariance.")

    header, d, v, keep, cmask = prepare(args)
    n = len(keep)
    wr = WeightReader(args.weights, header, n, raw=True)
    base = v["base_w"]

    preset = None if args.binning == "none" else args.binning
    bins, _ = parse_bins(args.bins, args.nbins, args.edges, preset)
    explicit = explicit_bin_vars(args.bins, args.edges)
    block_list = []
    for c in args.containment:
        for s in args.samples:
            for var in args.variables:
                e, _ = josh_bins(var, s, c, bins, preset, explicit)
                print(f"  binning {c}/{s}/{var}: {np.round(e, 4).tolist()}")
                block_list.append((s, var, e, c, cmask[c]))
    groups = ([block_list] if args.joint else [[b] for b in block_list])
    # BinVector selects with v["reco_pdg"], which holds the truth channel here
    bvs = [BinVector(g, v, keep) for g in groups]

    n_nom = [bv(base) for bv in bvs]
    stat_mc = [bv(base ** 2) for bv in bvs]

    E_dials = [dict() for _ in bvs]
    univ = [dict() for _ in bvs]
    univ_sig = [dict() for _ in bvs]
    ref = [dict() for _ in bvs]
    for dial in vary:
        info = header[dial]
        if dial not in wr.t.keys():
            print(f"  [skip] {dial}: not in SystWeights")
            continue
        if len(info["variations"]) <= 1:
            print(f"  [skip] {dial}: correction / single-valued dial, "
                  f"no universes")
            continue
        r_cv = wr.at_cv(dial)
        if r_cv is None:
            print(f"  [skip] {dial}: cv not in variations")
            continue
        w_ref = josh_weight(base, r_cv)
        vals, labs = [], []
        for s in sigmas:
            for sg in (-1, +1):
                r = wr.at_value(dial, info["cv"] + sg * s)
                if r is None:
                    print(f"  [info] {short(dial, args.short_labels)}: "
                          f"{sg * s:+g} sigma not available")
                    continue
                vals.append(josh_weight(base, r))
                labs.append(sg * s)
        if not vals:
            wr.cache.pop(dial, None)
            continue
        for k, bv in enumerate(bvs):
            nr = bv(w_ref)
            nu = [bv(w) for w in vals]
            ref[k][dial] = nr
            univ[k][dial] = nu
            univ_sig[k][dial] = labs
            E_dials[k][dial] = cov_from_universes(nr, nu)
        print(f"  {dial}: {len(vals)} universe(s)")
        wr.cache.pop(dial, None)

    if not E_dials[0]:
        sys.exit("No dial produced universes; nothing to do.")

    os.makedirs(args.outdir, exist_ok=True)
    prefix = (args.title or (args.group or "Selected dials")) + " [Josh]"
    for k, bv in enumerate(bvs):
        E_sys = sum(E_dials[k].values())
        E_stat = (np.diag(stat_mc[k]) if args.add_stat == "mc" else
                  np.diag(n_nom[k]) if args.add_stat == "poisson" else
                  np.zeros_like(E_sys))
        E_tot = E_sys + E_stat
        if args.joint:
            tag = "joint"
            what = " + ".join(f"{b['cont']}:{b['sample']}:{b['var']}"
                              for b in bv.blocks)
            sub = "joint: " + ", ".join(
                f"{CONT_LABEL[b['cont']]}{SHORT_SAMPLE[b['sample']]} "
                f"{SHORT_VAR[b['var']]}" for b in bv.blocks)
        else:
            b = bv.blocks[0]
            cpre = "" if b["cont"] == "all" else f"{b['cont']}_"
            tag = f"{cpre}{b['sample']}_{b['var']}"
            what = f"{cpre}{b['sample']}:{b['var']}"
            sub = (f"{CONT_LABEL[b['cont']]}{SAMPLE_LABEL[b['sample']]}, "
                   f"{SHORT_VAR[b['var']]}")
        stat_note = {"none": "", "mc": " + MC stat",
                     "poisson": " + Poisson stat"}[args.add_stat]
        miss = missing_bins(E_tot) | ~np.isfinite(n_nom[k])
        good = ~miss
        Eg = E_tot[np.ix_(good, good)]
        eig = np.linalg.eigvalsh(Eg) if good.any() else np.array([0.0])
        print(f"[{what}] {bv.n} bins, {len(E_dials[k])} dial(s); nominal = "
              f"{np.nansum(n_nom[k]):.6g}; min eigenvalue = {eig.min():.3g} "
              f"(rank {np.sum(eig > 1e-12 * max(eig.max(), 1e-300))})")

        stem = os.path.join(args.outdir, tag)
        fmt = args.format
        plot_matrix(E_tot, bv, f"{prefix}: covariance{stat_note}\n{sub}",
                    r"$E_{ij}$ [events$^2$]", f"{stem}_cov.{fmt}", args, "cov")
        plot_matrix(fractional(E_tot, n_nom[k]), bv,
                    f"{prefix}: fractional covariance{stat_note}\n{sub}",
                    r"$E_{ij}/(N_i N_j)$", f"{stem}_fraccov.{fmt}", args,
                    "frac")
        plot_matrix(correlation(E_tot), bv,
                    f"{prefix}: correlation{stat_note}\n{sub}",
                    r"$\rho_{ij}$", f"{stem}_corr.{fmt}", args, "corr")
        plot_breakdown(n_nom[k], E_tot, E_dials[k], bv,
                       f"{prefix}: fractional uncertainty{stat_note}\n{sub}",
                       f"{stem}_fracerr.{fmt}", args)
        tot_curves = []
        if args.band_universes:
            import matplotlib.pyplot as plt
            ranked = sorted(E_dials[k], key=lambda dd: -np.trace(E_dials[k][dd]))
            cmap = plt.get_cmap("tab10")
            for j, dd in enumerate(ranked[:args.band_universes]):
                tot_curves += universe_curves(
                    univ[k][dd], univ_sig[k][dd],
                    label_prefix=short(dd, args.short_labels) + " ",
                    color=cmap((j + 1) % 10), lw=1.2)
        plot_band(n_nom[k], E_tot, bv,
                  f"{prefix}: nominal $\\pm\\sqrt{{E_{{ii}}}}${stat_note}\n{sub}",
                  f"{stem}_band.{fmt}", args, curves=tot_curves,
                  band_label=r"$\pm\sqrt{E_{ii}}$ (sum of dials)",
                  cv_label="nominal (no dial)")
        save_outputs(stem, bv, n_nom[k], E_sys, E_stat, E_tot, E_dials[k],
                     univ[k], args, univ_sig[k])
        np.savez(stem + "_refs.npz",
                 **{f"ref__{dd}": r for dd, r in ref[k].items()})

        if args.per_dial_plots:
            ddir = os.path.join(args.outdir, "dials")
            os.makedirs(ddir, exist_ok=True)
            for dial, Ed in E_dials[k].items():
                lab = short(dial, args.short_labels)
                dstem = os.path.join(ddir, f"{tag}__{dial}")
                plot_matrix(Ed, bv, f"{lab} [Josh]: covariance\n{sub}",
                            r"$E^{(d)}_{ij}$ [events$^2$]",
                            f"{dstem}_cov.{fmt}", args, "cov")
                plot_matrix(correlation(Ed), bv,
                            f"{lab} [Josh]: correlation\n{sub}",
                            r"$\rho_{ij}$", f"{dstem}_corr.{fmt}", args, "corr")
                plot_band(ref[k][dial], Ed, bv,
                          f"{lab} [Josh]: dial at cv $\\pm\\sqrt{{E^{{(d)}}_{{ii}}}}$"
                          f"\n{sub}", f"{dstem}_band.{fmt}", args,
                          band_label=r"$\pm\sqrt{E^{(d)}_{ii}}$ (" + lab + ")",
                          curves=universe_curves(univ[k][dial],
                                                 univ_sig[k][dial]),
                          cv_label="dial at cv")

    print(f"Saved matrices, plots and .npz/.root files under {args.outdir}/")


if __name__ == "__main__":
    main()
