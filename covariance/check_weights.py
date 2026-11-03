#!/usr/bin/env python3
"""
check_weights.py

Optional diagnostic of the flux / cross-section / oscillation weights in the
CAF 'weights' tree. The plotting scripts use final_oscillated_w directly;
this shows how it relates to the other branches.

It reports:
  1. the TTree cycles present (e.g. weights;6 and weights;7) and whether they
     differ;
  2. per-branch statistics (N, NaN/inf, negatives, zeros, min/mean/max);
  3. which formula reproduces final_oscillated_w, among
        A = nue_w*osc_from_e_w + numu_w*osc_from_mu_w
        B = xsec * A
        plus a few other combinations;
  4. the relation of the 'xsec' branch to cafTree mc.nu.xsec / genweight /
     xsec_cvwgt, and of nue_w / numu_w to flux_nue / flux_numu;
  5. the total weighted event count per formula.

Usage:
    python check_weights.py                        # default CAF path
    python check_weights.py --caf other.root --max-events 500000
"""

import argparse
import sys

import numpy as np

try:
    import uproot
    import awkward as ak
except ImportError:
    sys.exit("needs uproot and awkward: pip install uproot awkward")

DEFAULT_CAF = "/exp/dune/data/users/pgranger/caf_new_sum.2.6M_weighted.root"
MC = "rec/mc/mc.nu/mc.nu."


def stats(name, x):
    x = np.asarray(x, float)
    fin = np.isfinite(x)
    xf = x[fin]
    if xf.size == 0:
        print(f"  {name:20s} N={x.size}  all non-finite")
        return
    print(f"  {name:20s} N={x.size:<9d} nonfinite={np.sum(~fin):<7d} "
          f"neg={np.sum(xf < 0):<7d} zero={np.sum(xf == 0):<8d} "
          f"min={xf.min():<11.4g} mean={xf.mean():<11.4g} "
          f"max={xf.max():.4g}")


def compare(label, a, b):
    """How well a reproduces b (relative difference on finite, b != 0)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    nz = ok & (b != 0)
    if nz.sum() == 0:
        print(f"  {label:42s} no comparable entries")
        return np.inf
    rel = np.abs(a[nz] - b[nz]) / np.abs(b[nz])
    zero_mismatch = np.sum(ok & (b == 0) & (a != 0))
    ratio = a[nz] / b[nz]
    print(f"  {label:42s} median|rel diff|={np.median(rel):.3g}  "
          f"max={rel.max():.3g}  frac(<1e-5)={np.mean(rel < 1e-5):.4f}  "
          f"ratio a/b: median={np.median(ratio):.4g}, "
          f"spread(p5-p95)={np.percentile(ratio, 5):.4g}-"
          f"{np.percentile(ratio, 95):.4g}"
          + (f"  [{zero_mismatch} entries with b=0, a!=0]"
             if zero_mismatch else ""))
    return np.median(rel)


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--caf", default=DEFAULT_CAF)
    p.add_argument("--max-events", type=int, default=None)
    args = p.parse_args()
    n = args.max_events

    f = uproot.open(args.caf)

    # 1. cycles -------------------------------------------------------------
    cycles = sorted((k for k in f.keys(cycle=True)
                     if k.split(";")[0] == "weights"),
                    key=lambda c: int(c.split(";")[1]))
    newest = max(cycles, key=lambda c: int(c.split(";")[1]))
    print(f"\n[1] 'weights' cycles in file: {cycles}   "
          f"(uproot's f['weights'] reads the newest: {newest})")
    trees = {c: f[c] for c in cycles}
    for c, t in trees.items():
        print(f"    {c}: {t.num_entries} entries, branches {t.keys()}")
    if len(cycles) > 1:
        ref = trees[cycles[-1]].arrays(library="np", entry_stop=n)
        for c in cycles[:-1]:
            other = trees[c]
            if other.num_entries != trees[cycles[-1]].num_entries:
                print(f"    {c} has a different number of entries "
                      f"(often an incomplete write; uproot uses the newest)")
                continue
            o = other.arrays(library="np", entry_stop=n)
            same = all(np.array_equal(ref[k], o[k], equal_nan=True)
                       for k in ref if k in o)
            print(f"    {c} identical to {cycles[-1]}: {same}")

    w = f["weights"].arrays(library="np", entry_stop=n)
    N = len(next(iter(w.values())))

    # 2. stats --------------------------------------------------------------
    print(f"\n[2] branch statistics (newest cycle, {N} entries)")
    for k, v in w.items():
        stats(k, v)

    # 3. which formula is final_oscillated_w ----------------------------------
    g = {k: np.asarray(v, float) for k, v in w.items()}
    A = g["nue_w"] * g["osc_from_e_w"] + g["numu_w"] * g["osc_from_mu_w"]
    F = g["flux_nue"] * g["osc_from_e_w"] + g["flux_numu"] * g["osc_from_mu_w"]
    cands = {
        "F = flux_nue*osc_e + flux_numu*osc_mu": F,
        "xsec * F": g["xsec"] * F,
        "A = nue_w*osc_e + numu_w*osc_mu": A,
        "B = xsec * A": g["xsec"] * A,
        "nue_w + numu_w (unoscillated)": g["nue_w"] + g["numu_w"],
        "xsec*(nue_w + numu_w)": g["xsec"] * (g["nue_w"] + g["numu_w"]),
        "weight * A": g["weight"] * A,
        "genie_weight * xsec * A": g["genie_weight"] * g["xsec"] * A,
    }
    print("\n[3] candidate formulas vs final_oscillated_w")
    best = min(cands, key=lambda k: compare(k, cands[k],
                                            g["final_oscillated_w"]))
    print(f"  -> closest: {best}")

    # 4. relations to cafTree / flux ---------------------------------------
    print("\n[4] relations")
    try:
        t = f["cafTree"]
        keys = set(t.keys(recursive=True))
        for leaf in ("xsec", "genweight", "xsec_cvwgt"):
            b = MC + leaf
            if b in keys:
                arr = t[b].array(library="ak", entry_stop=n)
                first = ak.to_numpy(ak.fill_none(ak.firsts(arr), np.nan))
                compare(f"weights.xsec vs mc.nu.{leaf}", g["xsec"], first)
    except Exception as e:
        print(f"  (could not read cafTree: {e})")
    compare("nue_w vs flux_nue", g["nue_w"], g["flux_nue"])
    compare("numu_w vs flux_numu", g["numu_w"], g["flux_numu"])
    compare("nue_w vs xsec*flux_nue", g["nue_w"], g["xsec"] * g["flux_nue"])
    compare("numu_w vs xsec*flux_numu", g["numu_w"], g["xsec"] * g["flux_numu"])
    compare("weight vs 1", g["weight"], np.ones(N))

    # 5. totals --------------------------------------------------------------
    print("\n[5] total weighted events")
    for k, v in cands.items():
        print(f"  {k:42s} {np.nansum(v):.6g}")
    print(f"  {'final_oscillated_w':42s} {np.nansum(g['final_oscillated_w']):.6g}")
    print("\nThe plotting scripts use final_oscillated_w directly as the base "
          "weight;\nthis check only shows how it relates to the other "
          "branches (diagnostic).")


if __name__ == "__main__":
    main()
