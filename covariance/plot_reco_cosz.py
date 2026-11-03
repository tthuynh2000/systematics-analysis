#!/usr/bin/env python3
"""
Plot reco cos(theta_Z) for nue-like and numu-like samples from DUNE CAF files.

--cosz mach3 (default), the MaCh3 atmospheric reco direction:
    nParticles = pandora particles of the first interaction, excluding the
                 unassociated-hits object (origRecoObjType == kHitCollection)
    useHits    = nParticles < 2 or !(Enu.calo < 1.3 GeV)      (NaN -> hits)
    d          = dir.calo   if useHits          (DirectionRecHits)
                 dir.heshw  else if nue-like    (DirectionRecNuEPfps)
                 dir.lngtrk else                (DirectionRecNuMuPfps)
    cosZ       = -d.y / |d|
--cosz pfp: the raw -dir.heshw.y for the nue-like sample and -dir.lngtrk.y
    for the numu-like sample (not normalised, no hits fallback);
    --show-both then overlays the other estimator as well (dashed).

The y axis is the weighted event count. Each event is weighted by

    w = base  x  prod_d  w_d(cv)

base (--base-weight, from the CAF 'weights' tree, newest cycle):
    flux_xsec (default): unoscillated xsec x flux of the event's own true
                         flavour (flux_nue for |pdg| = 12, flux_numu for 14,
                         0 otherwise)
    final:               final_oscillated_w
w_d(cv) is the raw stored response of dial d at its central value, from the
matching systematics file (SystWeights + systsHeader); the CV index is the
position of systsHeader/cv in systsHeader/variations, and correction dials
(one variation) use element 0. Raw responses: negative values are kept, and
a nonfinite one makes the bin it falls in missing (--clip-responses restores
negative -> 0, nonfinite -> 1). Dials named <provider>_b<N> (ZExpPCA b-scans)
share one zero-displacement response, which is applied once per provider
(--independent-b-scans multiplies each). Same conventions as
plot_syst_spectra.py. --unweighted gives plain counts; --no-dial-cv uses the
base weight only.

The CAF and systematics files must match entry by entry (same events, same
order). Give one systematics file per CAF file, in the same order.

Sample definition (pick one):
  --cvn FILE [FILE ...]   (+ --syst FILE [FILE ...])
      Classify events with CVN:
        --selection mach3 (default), as in MaCh3_DUNE SampleHandlerAtm:
            numu-like if cvn.numu > 0.56, else nue-like if cvn.nue > 0.55,
            else NC-like (not plotted).
        --selection analysis:
            pass_mu = cvn.numu > 0.56, pass_e = cvn.nue > 0.55;
            neither -> NC; only one -> that sample; both -> the larger score.
  --nue FILE [...] --numu FILE [...]   (+ --nue-syst / --numu-syst)
      Use already-selected files (e.g. *_nueselec.root / *_numuselec.root);
      every event in a file goes to that sample.
  With no sample option, --cvn DEFAULT_CAF with --syst DEFAULT_WEIGHTS is used.

Only the first Pandora interaction (pandora[0]) of each event is used, as in MaCh3.

Examples:
  python plot_reco_cosz.py                                   # default files
  python plot_reco_cosz.py --cvn caf_atm.root --syst syst_atm.root
  python plot_reco_cosz.py --nue A_nueselec.root --nue-syst A_nue_syst.root \\
                           --numu A_numuselec.root --numu-syst A_numu_syst.root
  python plot_reco_cosz.py --cvn caf.root --syst syst.root --bins 40 --density --out cosz.pdf
  python plot_reco_cosz.py --cvn caf.root --unweighted        # plain counts

Requires: pip install uproot awkward numpy matplotlib
"""
import argparse
import fnmatch
import re
import sys

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import uproot

DEFAULT_CAF = "/exp/dune/data/users/pgranger/caf_new_sum.2.6M_weighted.root"
DEFAULT_WEIGHTS = ("/exp/dune/data/users/tthuynh/"
                   "merged_new_systs_wsbn_hotfixed_full_header.root")

# Branch names are matched by suffix, so both the nested path
# ("rec/common/common.ixn.pandora/common.ixn.pandora.dir.heshw.y") and the flat
# name ("rec.common.ixn.pandora.dir.heshw.y") are found.
BR_HESHW_Y = "common.ixn.pandora.dir.heshw.y"
BR_LNGTRK_Y = "common.ixn.pandora.dir.lngtrk.y"
BR_CVN_NUMU = "common.ixn.pandora.nuhyp.cvn.numu"
BR_CVN_NUE = "common.ixn.pandora.nuhyp.cvn.nue"
BR_PDG = "mc.nu.pdg"
BR_ENU_CALO = "common.ixn.pandora.Enu.calo"
DIR_EST = ("calo", "heshw", "lngtrk")
BR_DIR = {f"{e}_{c}": f"common.ixn.pandora.dir.{e}.{c}"
          for e in DIR_EST for c in "xyz"}
HIT_COLLECTION = 3   # caf::RecoObjType::kHitCollection (duneanaobj SREnums.h)
SHARED_CV_RE = re.compile(r"^(.*)_b\d+$")   # <provider>_bN scans

CVN_NUMU_CUT = 0.56  # same hard-coded values as SampleHandlerAtm::ReturnSampleIdentifier
CVN_NUE_CUT = 0.55


def find_branch(tree, suffix):
    matches = [k for k in tree.keys(recursive=True) if k.endswith(suffix)]
    if not matches:
        sys.exit(f"Could not find a branch ending in '{suffix}' in tree '{tree.name}'.")
    return min(matches, key=len)  # shortest = the leaf itself, not a parent


def count_particles(tree, source="auto"):
    """nParticles of the first interaction (= NbPFPs - 1); NaN where there
    is no interaction. source: type (count non-hit-collection particles),
    npandora (part.npandora - 1), npfps (fd.hd.pandora.npfps - 1), or auto
    (type if readable, else npandora)."""
    if source in ("npandora", "npfps"):
        return _count_from(tree, source)
    keys = [k for k in tree.keys(recursive=True)
            if k.endswith("part.pandora.origRecoObjType")]
    try:
        if keys:
            typ = tree[min(keys, key=len)].array(library="ak")
        else:                        # unsplit object: take the field
            obj = [k for k in tree.keys(recursive=True)
                   if k.endswith("ixn.pandora.part.pandora")]
            if not obj:
                raise KeyError("no part.pandora branch")
            typ = tree[min(obj, key=len)].array(library="ak")["origRecoObjType"]
    except Exception as e:           # noqa: BLE001
        if source == "type":
            sys.exit(f"part.pandora.origRecoObjType could not be read ({e}); "
                     f"use --nparticles-from npandora / npfps or --cosz pfp")
        print(f"  [warn] particle types unreadable ({e}): using nParticles = "
              f"part.npandora - 1 (one unassociated-hits object assumed)")
        return _count_from(tree, "npandora")
    n = ak.sum(ak.firsts(typ, axis=1) != HIT_COLLECTION, axis=-1)
    return ak.to_numpy(ak.fill_none(n, np.nan)).astype(float)


def _count_from(tree, source):
    suffix = ("common.ixn.pandora.part.npandora" if source == "npandora"
              else "fd.hd.pandora.npfps")
    n = first_interaction(tree[find_branch(tree, suffix)].array(library="ak")) - 1
    try:                             # agreement of the two proxies
        a = first_interaction(tree[find_branch(tree, "common.ixn.pandora.part.npandora")]
                              .array(library="ak"))
        b = first_interaction(tree[find_branch(tree, "fd.hd.pandora.npfps")]
                              .array(library="ak"))
        ok = np.isfinite(a) & np.isfinite(b)
        print(f"  nParticles from {source}; check: part.npandora == "
              f"fd.hd.pandora.npfps for {100 * np.mean(a[ok] == b[ok]):.1f}% of events")
    except SystemExit:
        pass
    return n


def mach3_cosz(d, is_e):
    """MaCh3 reco cosZ; also returns which events used the hits direction."""
    use_hits = (d["n_particles"] < 2) | ~(d[BR_ENU_CALO] < 1.3)
    comp = {c: np.where(use_hits, d[f"calo_{c}"],
                        np.where(is_e, d[f"heshw_{c}"], d[f"lngtrk_{c}"]))
            for c in "xyz"}
    norm = np.sqrt(comp["x"] ** 2 + comp["y"] ** 2 + comp["z"] ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -comp["y"] / norm, use_hits


def first_interaction(arr):
    """Value for pandora[0]; NaN for events with no Pandora interaction."""
    if arr.ndim > 1:
        arr = ak.firsts(arr, axis=1)
    return ak.to_numpy(ak.fill_none(arr, np.nan)).astype(float)


# ----------------------------------------------------------------------------
# Event weights
# ----------------------------------------------------------------------------
def read_header(syst):
    arr = syst["systsHeader"].arrays(library="ak")
    names = [n.decode() if isinstance(n, bytes) else str(n)
             for n in ak.to_list(arr["name"])]
    cvs = ak.to_list(arr["cv"]) if "cv" in arr.fields else [0.0] * len(names)
    var = ak.to_list(arr["variations"])
    return {n: (float(c if c is not None else 0.0),
                [float(x) for x in (v or [])])
            for n, c, v in zip(names, cvs, var)}


def cv_index(cv, variations, tol=1e-6):
    for i, v in enumerate(variations):
        if abs(v - cv) < tol:
            return i
    return 0 if len(variations) <= 1 else None


def shared_key(name, names, independent):
    """CV-group key: all <provider>_bN scans of one provider share one."""
    m = None if independent else SHARED_CV_RE.match(name)
    if m and sum(1 for o in names if SHARED_CV_RE.match(o)
                 and SHARED_CV_RE.match(o).group(1) == m.group(1)) > 1:
        return m.group(1) + "_b*"
    return name


def dial_cv_product(syst_file, n_events, patterns, raw=True,
                    independent_b=False):
    """Per-event product of the selected dials' CV responses (each shared
    b-scan provider counted once)."""
    prod = np.ones(n_events)
    used, skipped, seen = 0, [], {}
    with uproot.open(syst_file) as syst:
        header = read_header(syst)
        t = syst["SystWeights"]
        if t.num_entries < n_events:
            sys.exit(f"{syst_file}: SystWeights has {t.num_entries} entries but "
                     f"the CAF has {n_events}; the files do not match.")
        if t.num_entries > n_events:
            print(f"  [warn] {syst_file}: SystWeights has {t.num_entries} "
                  f"entries, CAF has {n_events}; using the first {n_events}")
        branches = set(t.keys())
        for name, (cv, variations) in header.items():
            if not any(fnmatch.fnmatchcase(name, p) for p in patterns):
                continue
            idx = cv_index(cv, variations)
            if name not in branches or idx is None:
                skipped.append(name)
                continue
            w = t[name].array(library="ak", entry_stop=n_events)
            if w.ndim == 1:
                vals = ak.to_numpy(w).astype(float)
            else:
                k = ak.to_numpy(ak.num(w, axis=1))
                vals = np.ones(len(w))
                ok = k > idx
                vals[ok] = ak.to_numpy(w[ok][:, idx]).astype(float)
            if raw:
                vals = np.where(np.isfinite(vals), vals, np.nan)
            else:
                vals = np.clip(np.where(np.isfinite(vals), vals, 1.0), 0.0, None)
            used += 1
            key = shared_key(name, list(header), independent_b)
            if key in seen:            # same zero response: apply once
                if not np.allclose(vals, seen[key][1], equal_nan=True):
                    print(f"  [warn] {name}: CV response differs from "
                          f"{seen[key][0]}; using {seen[key][0]}'s once")
                continue
            seen[key] = (name, vals)
            prod = prod * vals
    shared = sorted(k for k in seen if k.endswith("_b*"))
    print(f"  CV weight from {used} dial(s) ({len(seen)} independent "
          f"factor(s)) in {syst_file}"
          + (f"; shared b-scan zero response applied once: {', '.join(shared)}"
             if shared else "")
          + (f"; skipped {len(skipped)} (cv not in variations or no branch): "
             f"{', '.join(skipped)}" if skipped else ""))
    nonfin = int(np.sum(~np.isfinite(prod)))
    if nonfin:
        print(f"  {nonfin} events with a nonfinite CV response "
              f"(their bins become missing)")
    return prod


def event_weights(caf, n_events, syst_file, pdg, args):
    if args.unweighted:
        return np.ones(n_events)
    wt = caf["weights"]          # uproot picks the newest cycle
    if wt.num_entries != n_events:
        sys.exit(f"'weights' tree has {wt.num_entries} entries, cafTree has "
                 f"{n_events}.")
    g = lambda b: wt[b].array(library="np").astype(float)  # noqa: E731
    if args.base_weight == "final":
        base = g("final_oscillated_w")
    else:                        # unoscillated xsec x true-flavour flux
        apdg = np.abs(pdg)
        base = g("xsec") * np.select([apdg == 12, apdg == 14],
                                     [g("flux_nue"), g("flux_numu")], 0.0)
    base = np.nan_to_num(base, nan=0.0)
    if args.no_dial_cv:
        return base
    if syst_file is None:
        sys.exit("Dial CV weights need a systematics file per CAF file "
                 "(--syst / --nue-syst / --numu-syst), or use --no-dial-cv "
                 "or --unweighted.")
    return base * dial_cv_product(syst_file, n_events, args.cv_dials,
                                  raw=not args.clip_responses,
                                  independent_b=args.independent_b_scans)


# ----------------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------------
def read_files(files, syst_files, tree_name, need_cvn, args):
    if syst_files is not None and len(syst_files) != len(files):
        sys.exit(f"Got {len(files)} CAF file(s) but {len(syst_files)} "
                 f"systematics file(s); give one per CAF file, same order.")
    suffixes = [BR_HESHW_Y, BR_LNGTRK_Y] + ([BR_CVN_NUMU, BR_CVN_NUE] if need_cvn else [])
    extra = ([BR_ENU_CALO] + list(BR_DIR.values())) if args.cosz == "mach3" else []
    out = {s: [] for s in suffixes + extra + ["weight", "n_particles"]}
    for i, fname in enumerate(files):
        with uproot.open(fname) as f:
            tree = f[tree_name]
            names = {s: find_branch(tree, s) for s in suffixes + extra}
            arrays = tree.arrays(list(names.values()), library="ak")
            for s, b in names.items():
                out[s].append(first_interaction(arrays[b]))
            if args.cosz == "mach3":
                out["n_particles"].append(count_particles(tree, args.nparticles_from))
            n = tree.num_entries
            pdg = (first_interaction(tree[find_branch(tree, BR_PDG)]
                                     .array(library="ak"))
                   if not args.unweighted and args.base_weight == "flux_xsec"
                   else None)
            out["weight"].append(event_weights(
                f, n, syst_files[i] if syst_files else None, pdg, args))
        print(f"read {fname}")
    d = {s: np.concatenate(v) if v else np.array([]) for s, v in out.items()}
    for key, br in BR_DIR.items():          # short names for mach3_cosz
        if br in d:
            d[key] = d[br]
    return d


def sample_cosz(d, mask, is_e, args):
    """Per-sample arrays to plot: 'mach3' (MaCh3 cosZ) or the raw pfp ones."""
    out = {"weight": d["weight"][mask], "n_events": int(mask.sum())}
    if args.cosz == "mach3":
        sub = {k: d[k][mask] for k in list(BR_DIR) + [BR_ENU_CALO, "n_particles"]}
        cz, hits = mach3_cosz(sub, is_e)
        out["mach3"], out["hits_frac"] = cz, float(np.mean(hits)) if len(hits) else 0.0
    out["heshw"], out["lngtrk"] = -d[BR_HESHW_Y][mask], -d[BR_LNGTRK_Y][mask]
    return out


def clean(x, w):
    """Drop NaN and CAF default/sentinel values (anything outside [-1, 1])."""
    keep = np.isfinite(x) & (np.abs(x) <= 1.0)
    return x[keep], w[keep]


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def whist(x, edges, w):
    """Weighted histogram ([lo, hi), last bin closed) via bincount: a
    nonfinite weight makes only its own bin missing (np.histogram would
    spread the NaN to the following bins)."""
    nb = len(edges) - 1
    idx = np.searchsorted(edges, x, side="right") - 1
    idx[x == edges[-1]] = nb - 1
    ok = (idx >= 0) & (idx < nb)
    return np.bincount(idx[ok], weights=w[ok], minlength=nb)


def plot(samples, bins, density, out, args):
    edges = np.linspace(-1, 1, bins + 1)
    fig, axes = plt.subplots(1, len(samples), figsize=(6 * len(samples), 4.5), squeeze=False)
    for ax, (label, d) in zip(axes[0], samples.items()):
        used_by_mach3 = "heshw" if label.startswith("nue") else "lngtrk"
        if args.cosz == "mach3":
            estimators = [("mach3", f"MaCh3: calo {100 * d['hits_frac']:.0f}%, "
                           f"{used_by_mach3} {100 * (1 - d['hits_frac']):.0f}%")]
        else:
            estimators = [("heshw", r"$-\hat{d}_y$ (heshw, shower)"),
                          ("lngtrk", r"$-\hat{d}_y$ (lngtrk, track)")]
            if not args.show_both:     # only the estimator MaCh3 uses
                estimators = [e for e in estimators if e[0] == used_by_mach3]
        for key, name in estimators:
            vals, w = clean(d[key], d["weight"])
            h = whist(vals, edges, w)
            if density:
                h = h / (np.nansum(h) * np.diff(edges))
            nmiss = int(np.sum(~np.isfinite(h)))
            ax.stairs(h, edges,
                      color={"heshw": "tab:blue", "lngtrk": "tab:orange"}.get(
                          key, "tab:blue" if label.startswith("nue") else "tab:orange"),
                      linewidth=2.0 if key in (used_by_mach3, "mach3") else 1.2,
                      linestyle="-" if key in (used_by_mach3, "mach3") else "--",
                      label=f"{name}  N={len(vals)}"
                            + ("" if args.unweighted else f", Σw={np.nansum(w):.4g}")
                            + (f", {nmiss} bin(s) missing" if nmiss else "")
                            + ("  [MaCh3]" if args.cosz == "pfp" and args.show_both
                               and key == used_by_mach3 else ""))
        ax.set_title(f"{label}  ({d['n_events']} events)")
        ax.set_xlabel(r"reco $\cos\theta_Z$")
        if density:
            ax.set_ylabel("normalised")
        elif args.unweighted:
            ax.set_ylabel("events")
        else:
            bw = ("final_oscillated_w" if args.base_weight == "final"
                  else r"xsec $\times$ flux (unosc.)")
            ax.set_ylabel("weighted events\n(" + bw
                          + ("" if args.no_dial_cv
                             else r" $\times\ \prod_d w_d^{\rm CV}$") + ")")
        ax.set_xlim(-1, 1)
        ax.legend(fontsize=8, loc="upper center")
    fig.tight_layout()
    fig.savefig(out)
    print(f"saved {out}")


def classify(d, selection):
    mu, e = d[BR_CVN_NUMU], d[BR_CVN_NUE]
    pass_mu, pass_e = mu > CVN_NUMU_CUT, e > CVN_NUE_CUT
    if selection == "mach3":
        numu = pass_mu
        nue = ~numu & pass_e
    else:  # analysis: if both pass, the larger score wins
        both = pass_mu & pass_e
        numu = (pass_mu & ~pass_e) | (both & (mu > e))
        nue = (pass_e & ~pass_mu) | (both & ~(mu > e))
    return nue, numu


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                allow_abbrev=False)
    p.add_argument("--cvn", nargs="+", help="CAF file(s) to split into nue/numu-like with CVN cuts")
    p.add_argument("--syst", nargs="+", help="systematics file(s) matching --cvn, same order")
    p.add_argument("--nue", nargs="+", help="pre-selected nue-like file(s)")
    p.add_argument("--nue-syst", nargs="+", help="systematics file(s) matching --nue")
    p.add_argument("--numu", nargs="+", help="pre-selected numu-like file(s)")
    p.add_argument("--numu-syst", nargs="+", help="systematics file(s) matching --numu")
    p.add_argument("--selection", choices=["mach3", "analysis"], default="mach3",
                   help="CVN classification for --cvn (see top of file)")
    p.add_argument("--cv-dials", nargs="+", default=["*"],
                   help="dials (globs) whose CV weight multiplies the event weight (default: all)")
    p.add_argument("--base-weight", choices=["flux_xsec", "final"], default="flux_xsec",
                   help="flux_xsec: unoscillated xsec x true-flavour flux (default); "
                        "final: final_oscillated_w")
    p.add_argument("--clip-responses", action="store_true",
                   help="old dial handling: negative -> 0, nonfinite -> 1")
    p.add_argument("--independent-b-scans", action="store_true",
                   help="multiply the CV response of every <provider>_bN scan")
    p.add_argument("--no-dial-cv", action="store_true",
                   help="weight by the base weight only")
    p.add_argument("--unweighted", action="store_true",
                   help="plain event counts (no weights)")
    p.add_argument("--cosz", choices=["mach3", "pfp"], default="mach3",
                   help="mach3: MaCh3 reco direction (calo if <2 particles or "
                        "Enu.calo >= 1.3/NaN, else heshw (nue-like) / lngtrk), "
                        "normalised; pfp: raw -heshw.y (nue-like) / -lngtrk.y")
    p.add_argument("--nparticles-from", choices=["auto", "type", "npandora", "npfps"],
                   default="auto",
                   help="nParticles for --cosz mach3: type (non-hit-collection "
                        "particles), part.npandora - 1, fd.hd.pandora.npfps - 1; "
                        "auto = type if readable, else npandora")
    p.add_argument("--show-both", action="store_true",
                   help="with --cosz pfp: also overlay the other estimator (dashed)")
    p.add_argument("--tree", default="cafTree", help="tree name (default: cafTree)")
    p.add_argument("--bins", type=int, default=20)
    p.add_argument("--density", action="store_true", help="area-normalise histograms")
    p.add_argument("--out", default="reco_cosz.png")
    a = p.parse_args()

    if not (a.cvn or a.nue or a.numu):
        a.cvn = [DEFAULT_CAF]
        if a.syst is None:
            a.syst = [DEFAULT_WEIGHTS]
        print(f"no sample option given: using --cvn {DEFAULT_CAF} --syst {a.syst[0]}")

    samples = {}
    if a.cvn:
        d = read_files(a.cvn, a.syst, a.tree, need_cvn=True, args=a)
        nue, numu = classify(d, a.selection)
        for label, mask in [("nue-like", nue), ("numu-like", numu)]:
            samples[label] = sample_cosz(d, mask, label == "nue-like", a)
        nc = ~nue & ~numu
        print(f"[{a.selection}] nue-like: {nue.sum()}, numu-like: {numu.sum()}, "
              f"NC-like (not plotted): {nc.sum()}")
        if not a.unweighted:
            print(f"  sum of weights (finite)  nue-like: {np.nansum(d['weight'][nue]):.6g}, "
                  f"numu-like: {np.nansum(d['weight'][numu]):.6g}, "
                  f"NC-like: {np.nansum(d['weight'][nc]):.6g}")
    else:
        for label, files, systs in [("nue-like", a.nue, a.nue_syst),
                                    ("numu-like", a.numu, a.numu_syst)]:
            if files:
                d = read_files(files, systs, a.tree, need_cvn=False, args=a)
                samples[label] = sample_cosz(d, np.ones(len(d[BR_HESHW_Y]), bool),
                                             label == "nue-like", a)

    plot(samples, a.bins, a.density, a.out, a)


if __name__ == "__main__":
    main()
