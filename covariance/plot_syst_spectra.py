#!/usr/bin/env python3
"""
plot_syst_spectra.py

Reco and true spectra for each cross-section dial at its cv and at +/- n
sigma, with a ratio panel. The y axis is events / bin width by default.
syst_lib.py (plotting, binning, dial reading) and plot_cv_weights.py (dial
groups) must sit in the same folder.

Samples are TRUE interaction channels, not CVN selections
----------------------------------------------------------
cc_nue  : true CC, |pdg| = 12 (nu_e and nu_e-bar)      "electron channel"
cc_numu : true CC, |pdg| = 14 (nu_mu and nu_mu-bar)    "muon channel"
nc      : true NC, |pdg| = 12 or 14
cc_nue+cc_numu, all : sums. nu_tau is not a supported channel (zero flux).

Common selection (each cut can be dropped with --skip-cuts)
-----------------------------------------------------------
one_nu      exactly one MC neutrino in the entry
one_ixn     exactly one pandora interaction
channel     supported truth channel (above)
base        nominal weight xsec * matching-flavour flux finite and > 0
energy      true E in [--emin, --emax] = [0.1, 100] GeV
fv          vertex in the fiducial volume (--fv box; off unless given)
containment valid containment minimum: 0 <= walldist_min <= --walldist-max
true_dir    valid true direction (|p| > 0)
reco_dir    valid reco direction (finite, |d| > 0; horizontal is valid)
Each observable additionally has its own finite-value mask.

Containment (needs make_part_info.py -> part_info.npz)
------------------------------------------------------
walldist_min = minimum particle walldist [cm] of the interaction, excluding
the unassociated-hits object (origRecoObjType == 3); NaN walls are ignored.
FC: walldist_min > 10 cm; PC: 0 <= walldist_min <= 10 cm (inclusive);
missing, negative or > 10000 cm: unknown (removed by the containment cut).

Reconstruction
--------------
nParticles = pandora particles excluding the hit collection (part_info.npz)
direction  = dir.calo  if NC, or nParticles < 2, or !(Enu.calo < 1.3 GeV)
             dir.heshw for the electron channel, dir.lngtrk for muon
cosZ       = -d.y / |d|
reco E     = Enu.e_calo (electron channel), Enu.lep_calo (muon channel)
reco Ehad  = Enu.e_had (both)            reco Elep = E - Ehad
reco y     = Ehad / E if E > 0 else 0   (values outside [0, 1] are kept)
NC         : reco E = reco Ehad = Enu.calo (visible-energy proxy);
             reco Elep and reco y are unavailable (NaN)
--reco-energy matched_range: muon channel uses Enu.mu_range / Enu.mu_had,
             valid only when both are finite and mu_range > 0
true E = mc.nu.E; true Ehad = q0; true Elep = E - q0;
true y = mc.nu.inelasticity; true cosZ = -p.y / |p|

Weights (one-dial comparisons)
------------------------------
nominal   = xsec * flux_nue (|pdg| = 12) or xsec * flux_numu (|pdg| = 14),
            unoscillated, no other factor, not divided by the number of files
response  = stored dial weight; nonfinite -> 1; clipped to [-9999, 9999]
weight    = nominal * response, and weights <= 0 set to 0
Only the varied dial is applied (no other dial's CV; shared CV factors are
not multiplied). Black curve = the dial at its cv; "nominal" = no dial.

Example
-------
python make_part_info.py                       # once, with ROOT
python plot_syst_spectra.py --group CCQE_Axial_Deut_PCA \\
    --containment fc pc --samples cc_nue cc_numu \\
    --variables reco_E reco_y reco_cosz --uniform-axis
"""

import argparse
import os
import sys

import numpy as np
import matplotlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from syst_lib import (  # noqa: E402
        GROUPS, VARIABLES, BINNING_PRESETS, SAMPLES, SAMPLE_LABEL, CONT_LABEL,
        SIGMA_COLORS, DEFAULT_CAF, DEFAULT_WEIGHTS, load_caf, read_header,
        WeightReader, match_dials, parse_bins, explicit_bin_vars, sample_bins,
        hist, draw, short, first_or_nan, MC, PAN)
except ImportError as e:
    sys.exit(f"plot_syst_spectra.py needs syst_lib.py and "
             f"plot_cv_weights.py in the same folder: {e}")

import uproot
import awkward as ak

# ---------------------------------------------------------------------------
# Truth channels (codes chosen to match plot_syst_spectra's NUE/NUMU/NC, so
# shared helpers that look up samples by code keep working)
# ---------------------------------------------------------------------------
CC_NUE, CC_NUMU, NC_TRUE, OTHER = 12, 14, 0, -1
CHANNEL_SAMPLES = {"cc_nue": (CC_NUE,), "cc_numu": (CC_NUMU,), "nc": (NC_TRUE,),
                "cc_nue+cc_numu": (CC_NUE, CC_NUMU),
                "all": (CC_NUE, CC_NUMU, NC_TRUE)}
CHANNEL_SAMPLE_LABEL = {
    "cc_nue": r"true CC $\nu_e$ ($\bar\nu_e$)",
    "cc_numu": r"true CC $\nu_\mu$ ($\bar\nu_\mu$)",
    "nc": "true NC",
    "cc_nue+cc_numu": r"true CC $\nu_e$ + CC $\nu_\mu$",
    "all": r"true CC $\nu_e$ + CC $\nu_\mu$ + NC",
}
# make the shared helpers / plot labels know these names (this process only)
SAMPLES.update(CHANNEL_SAMPLES)
SAMPLE_LABEL.update(CHANNEL_SAMPLE_LABEL)
# FC/PC binning table is keyed by nue / numu
BIN_SAMPLE = {"cc_nue": "nue", "cc_numu": "numu"}
# True-energy definitions
VARIABLES["true_Elep"] = (r"True $E_{\rm lep} = E_\nu - q_0$ [GeV]",
                          *VARIABLES["true_Elep"][1:])
VARIABLES["true_Ehad"] = (r"True $E_{\rm had} = q_0$ [GeV]",
                          *VARIABLES["true_Ehad"][1:])
VARIABLES["reco_Ehad"] = (r"Reconstructed $E_{\rm had}$ (NC: $E_{\rm calo}$) [GeV]",
                          *VARIABLES["reco_Ehad"][1:])
VARIABLES["reco_E"] = (r"Reconstructed $E_\nu$ (NC: $E_{\rm calo}$) [GeV]",
                       *VARIABLES["reco_E"][1:])
VARIABLES["reco_y"] = (r"Reconstructed $y = E_{\rm had}/E_\nu$",
                       *VARIABLES["reco_y"][1:])

CONTAINMENTS = ("all", "fc", "pc")
CUTS = ("one_nu", "one_ixn", "channel", "base", "energy", "fv", "containment",
        "true_dir", "reco_dir")
RESPONSE_BOUND = 9999.0


# ---------------------------------------------------------------------------
# Reading the extra branches
# ---------------------------------------------------------------------------
def load_extra(path, max_events):
    """Branches needed beyond load_caf()."""
    wanted = {
        "q0": MC + "q0", "momx": MC + "momentum.x", "momz": MC + "momentum.z",
        "tvtx_x": MC + "vtx.x", "tvtx_y": MC + "vtx.y", "tvtx_z": MC + "vtx.z",
        "rvtx_x": PAN + "vtx.x", "rvtx_y": PAN + "vtx.y", "rvtx_z": PAN + "vtx.z",
        "mu_range": PAN + "Enu.mu_range",
    }
    with uproot.open(path) as f:
        t = f["cafTree"]
        keys = t.keys(recursive=True, full_paths=True)
        by_leaf = {}
        for k in keys:
            by_leaf.setdefault(k.split("/")[-1], k)
        out = {}
        for name, b in wanted.items():
            br = b if b in keys else by_leaf.get(b.split("/")[-1])
            if br is None:
                print(f"  [warn] branch {b} not found; {name} unavailable")
                continue
            out[name] = first_or_nan(t[br].array(library="ak",
                                                 entry_stop=max_events),
                                     name, False)
        e_br = MC + "E" if MC + "E" in keys else by_leaf.get("mc.nu.E")
        out["n_nu"] = ak.to_numpy(ak.num(t[e_br].array(
            library="ak", entry_stop=max_events), axis=1)).astype(float)
    return out


# ---------------------------------------------------------------------------
# Reconstruction, truth and nominal weight
# ---------------------------------------------------------------------------
def truth_channel(d):
    apdg = np.abs(d["pdg"])
    iscc = d["iscc"]
    flav = np.isin(apdg, (12, 14))
    return np.select([(iscc > 0) & (apdg == 12), (iscc > 0) & (apdg == 14),
                      (iscc == 0) & flav],
                     [CC_NUE, CC_NUMU, NC_TRUE], default=OTHER)


def derive(d, x, args):
    ch = truth_channel(d)
    e_ch, mu_ch, nc = ch == CC_NUE, ch == CC_NUMU, ch == NC_TRUE
    nan = np.full(len(ch), np.nan)

    # ---- reco energies
    if args.reco_energy == "matched_range":
        mr = x.get("mu_range", nan)
        mu_tot = np.where(np.isfinite(mr) & (mr > 0) & np.isfinite(d["mu_had"]),
                          mr, np.nan)
        mu_had = np.where(np.isfinite(mu_tot), d["mu_had"], np.nan)
    else:                                   # Pierre/Dan/Miles convention
        mu_tot, mu_had = d["lep_calo"], d["e_had"]
    tot = np.select([e_ch, mu_ch], [d["e_calo"], mu_tot], default=np.nan)
    had = np.select([e_ch, mu_ch], [d["e_had"], mu_had], default=np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        y_reco = np.where(tot > 0, had / tot, 0.0)
    y_reco = np.where(np.isfinite(tot) & np.isfinite(had), y_reco, np.nan)
    reco_E = np.where(nc, d["enu_calo"], tot)
    reco_Ehad = np.where(nc, d["enu_calo"], had)
    reco_Elep = np.where(nc, np.nan, tot - had)
    reco_y = np.where(nc, np.nan, y_reco)

    # ---- reco direction (hybrid; NC always hits)
    npart = d.get("n_particles")
    use_hits = nc | (npart < 2) | ~(d["enu_calo"] < args.hits_ecut)
    comp = {c: np.where(use_hits, d["calo_" + c],
                        np.where(e_ch, d["heshw_" + c], d["lngtrk_" + c]))
            for c in "xyz"}
    norm = np.sqrt(comp["x"] ** 2 + comp["y"] ** 2 + comp["z"] ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        reco_cosz = -comp["y"] / norm
    reco_dir_ok = np.isfinite(norm) & (norm > 0) & np.isfinite(reco_cosz)

    # ---- truth
    E, q0 = d["E"], x.get("q0", nan)
    p = np.sqrt(x.get("momx", nan) ** 2 + d["momy"] ** 2
                + x.get("momz", nan) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        true_cosz = -d["momy"] / p
    true_dir_ok = np.isfinite(p) & (p > 0) & np.isfinite(true_cosz)

    # ---- nominal weight
    apdg = np.abs(d["pdg"])
    flux = np.select([apdg == 12, apdg == 14],
                     [d["w_flux_nue"], d["w_flux_numu"]], default=0.0)
    base = d["w_xsec"] * flux

    return {
        "reco_pdg": ch,                     # used by shared sample masks
        "channel": ch,
        "reco_E": reco_E, "reco_Ehad": reco_Ehad, "reco_Elep": reco_Elep,
        "reco_y": reco_y, "reco_cosz": reco_cosz,
        "true_E": E, "true_Ehad": q0, "true_Elep": E - q0,
        "true_y": d["y"], "true_cosz": true_cosz,
        "base_w": np.where(np.isfinite(base) & (base > 0), base, 0.0),
        "_base_raw": base, "_reco_dir_ok": reco_dir_ok,
        "_true_dir_ok": true_dir_ok, "_use_hits": use_hits,
    }


# ---------------------------------------------------------------------------
# Selection and containment
# ---------------------------------------------------------------------------
def containment_valid(wd, wmax):
    return np.isfinite(wd) & (wd >= 0) & (wd <= wmax)


def containment_mask(cont, wd, cut, wmax):
    ok = containment_valid(wd, wmax)
    if cont == "fc":
        return ok & (wd > cut)
    if cont == "pc":
        return ok & (wd <= cut)
    return ok


def common_selection(d, x, v, args):
    """Cumulative cut flow; returns the final mask."""
    n = len(v["channel"])
    if args.fv is not None:
        pre = "tvtx_" if args.fv_vertex == "true" else "rvtx_"
        x0, x1, y0, y1, z0, z1 = args.fv
        vx, vy, vz = (x.get(pre + c, np.full(n, np.nan)) for c in "xyz")
        fv = ((vx >= x0) & (vx <= x1) & (vy >= y0) & (vy <= y1)
              & (vz >= z0) & (vz <= z1))
    else:
        fv = np.ones(n, bool)
    wd = d["walldist_min"]
    E = v["true_E"]
    masks = {
        "one_nu": x["n_nu"] == 1,
        "one_ixn": d["n_reco"] == 1,
        "channel": v["channel"] != OTHER,
        "base": np.isfinite(v["_base_raw"]) & (v["_base_raw"] > 0),
        "energy": np.isfinite(E) & (E >= args.emin) & (E <= args.emax),
        "fv": fv,
        "containment": containment_valid(wd, args.walldist_max),
        "true_dir": v["_true_dir_ok"],
        "reco_dir": v["_reco_dir_ok"],
    }
    keep = np.ones(n, bool)
    print("  cut flow (events remaining):")
    print(f"    {'all entries':14s} {n}")
    for c in CUTS:
        if c in args.skip_cuts:
            print(f"    {c:14s} skipped")
            continue
        if c == "fv" and args.fv is None:
            print(f"    {c:14s} not applied (no --fv box given)")
            continue
        keep &= masks[c]
        print(f"    {c:14s} {keep.sum()}")
    for s, codes in (("cc_nue", (CC_NUE,)), ("cc_numu", (CC_NUMU,)),
                     ("nc", (NC_TRUE,))):
        m = keep & np.isin(v["channel"], codes)
        print(f"    -> {s:8s} {m.sum():8d} events, nominal sum "
              f"{v['base_w'][m].sum():.6g}")
    return keep


def check_part_info(d, args):
    if d.get("walldist_min") is None:
        sys.exit("The selection needs the containment minimum and the exact "
                 "particle count: run make_part_info.py (with ROOT) to make "
                 f"{args.part_info}, or pass --part-info <file>")


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------
def dial_response(r):
    """Stored response -> nonfinite to 1, clipped to [-9999, 9999]."""
    r = np.where(np.isfinite(r), r, 1.0)
    return np.clip(r, -RESPONSE_BOUND, RESPONSE_BOUND)


def dial_weight(base, r):
    """nominal * response, non-positive totals set to 0."""
    w = base * dial_response(r)
    return np.where(w > 0, w, 0.0)


def channel_bins(var, sample, cont, bins, preset, explicit):
    return sample_bins(var, BIN_SAMPLE.get(sample, sample), cont, bins,
                       preset, explicit)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def add_selection_args(p):
    g = p.add_argument_group("selection / reconstruction")
    g.add_argument("--samples", nargs="*", default=["cc_nue", "cc_numu", "nc"],
                   choices=list(CHANNEL_SAMPLES),
                   help="TRUE interaction channels")
    g.add_argument("--containment", nargs="+", choices=CONTAINMENTS,
                   default=["all"],
                   help="fc: walldist_min > --walldist-cut; pc: 0 <= "
                        "walldist_min <= cut; all: FC + PC")
    g.add_argument("--walldist-cut", type=float, default=10.0)
    g.add_argument("--walldist-max", type=float, default=10000.0,
                   help="walldist_min above this is 'unknown'")
    g.add_argument("--part-info", default="part_info.npz",
                   help="Output of make_part_info.py (required)")
    g.add_argument("--emin", type=float, default=0.1, help="true E min [GeV]")
    g.add_argument("--emax", type=float, default=100.0, help="true E max [GeV]")
    g.add_argument("--fv", nargs=6, type=float, default=None,
                   metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
                   help="Vertex fiducial box [cm] (default: not applied)")
    g.add_argument("--fv-vertex", choices=["true", "reco"], default="true",
                   help="Vertex used for --fv")
    g.add_argument("--skip-cuts", nargs="*", default=[], choices=CUTS,
                   help="Drop these common-selection cuts")
    g.add_argument("--reco-energy", choices=["pierre", "matched_range"],
                   default="pierre",
                   help="pierre: e_calo / lep_calo with e_had for both; "
                        "matched_range: muon channel uses mu_range / mu_had")
    g.add_argument("--hits-ecut", type=float, default=1.3,
                   help="use dir.calo when !(Enu.calo < this) [GeV]")


def add_dial_args(p):
    g = p.add_argument_group("dials")
    g.add_argument("--list", action="store_true",
                   help="List dials, cv and variations, then exit")
    g.add_argument("--group", choices=sorted(GROUPS) if GROUPS else None)
    g.add_argument("--dials", nargs="*", default=[])
    g.add_argument("--regex", nargs="*", default=[])
    g.add_argument("--exclude", nargs="*", default=[])
    return g


def add_binning_args(p):
    g = p.add_argument_group("variables & binning")
    g.add_argument("--variables", nargs="*",
                   default=["reco_E", "reco_y", "reco_cosz"],
                   choices=list(VARIABLES))
    g.add_argument("--binning", choices=sorted(BINNING_PRESETS) + ["none"],
                   default="atm_fcpc")
    g.add_argument("--edges", nargs="*", default=[])
    g.add_argument("--bins", nargs="*", default=[])
    g.add_argument("--nbins", type=int, default=None)
    return g


def parse_args():
    p = argparse.ArgumentParser(
        description="Spectra with dial variations (truth-channel samples, "
                    "one-dial weights).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False)
    p.add_argument("--caf", default=DEFAULT_CAF)
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--max-events", type=int, default=None)
    g = add_dial_args(p)
    g.add_argument("--sigmas", nargs="*", type=float, default=[1, 2, 3])
    g.add_argument("--only-up", action="store_true")
    g.add_argument("--only-down", action="store_true")
    g.add_argument("--show-nominal", action="store_true",
                   help="Also draw the nominal weight (no dial applied)")
    add_selection_args(p)
    g = add_binning_args(p)
    g.add_argument("--uniform-axis", action="store_true")
    g.add_argument("--per-width", dest="per_width", action="store_true",
                   default=True,
                   help="y axis = events / bin width (default)")
    g.add_argument("--no-per-width", dest="per_width", action="store_false",
                   help="y axis = events per bin")
    g.add_argument("--logx", action="store_true")

    g = p.add_argument_group("style")
    g.add_argument("--title", default=None)
    g.add_argument("--ylabel", default=None)
    g.add_argument("--title-fontsize", type=float, default=13)
    g.add_argument("--label-fontsize", type=float, default=12)
    g.add_argument("--logy", action="store_true")
    g.add_argument("--linewidth", type=float, default=1.5)
    g.add_argument("--stat-band", action="store_true")
    g.add_argument("--ratio-range", nargs=2, type=float, default=None)
    g.add_argument("--ratio-logy", action="store_true")
    g.add_argument("--no-legend", dest="legend", action="store_false")
    g.add_argument("--legend-loc", default="best")
    g.add_argument("--legend-fontsize", type=float, default=9)
    g.add_argument("--legend-ncol", type=int, default=1)
    g.add_argument("--short-labels", action="store_true")
    g.add_argument("--grid-lines", action="store_true")
    g.add_argument("--figsize", nargs=2, type=float, default=(8, 7))

    g = p.add_argument_group("output")
    g.add_argument("--outdir", default="syst_spectra")
    g.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    g.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()
    if args.per_width and args.ylabel is None:
        args.ylabel = "Events / bin width"
    return args


def prepare(args):
    """Read everything and apply the common selection."""
    header = read_header(args.weights)
    print("Reading CAF ...")
    d = load_caf(args.caf, args.max_events, args.part_info)
    check_part_info(d, args)
    x = load_extra(args.caf, args.max_events)
    v = derive(d, x, args)
    keep = common_selection(d, x, v, args)
    wd = d["walldist_min"]
    cmask = {c: containment_mask(c, wd, args.walldist_cut,
                                      args.walldist_max)
             for c in args.containment}
    for c in args.containment:
        print(f"  {c}: {np.sum(keep & cmask[c])} selected events")
    return header, d, v, keep, cmask


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
        sys.exit("Choose dials with --group, --dials or --regex (or --list).")
    vary = match_dials(header, patterns, args.regex, args.exclude)
    if not vary:
        sys.exit("No dials matched.")

    header, d, v, keep, cmask = prepare(args)
    n = len(keep)
    wr = WeightReader(args.weights, header, n, raw=True)
    base = v["base_w"]

    preset = None if args.binning == "none" else args.binning
    bins, custom = parse_bins(args.bins, args.nbins, args.edges, preset)
    explicit = explicit_bin_vars(args.bins, args.edges)
    args.logx_vars = ([vv for vv in args.variables if not vv.endswith("cosz")]
                      if args.logx else [])

    sigmas = sorted({abs(s) for s in args.sigmas if s != 0})
    signs = [+1] if args.only_up else [-1] if args.only_down else [+1, -1]
    blocks = [(c, s) for c in args.containment for s in args.samples]
    reported = set()
    n_plots = 0
    for dial in vary:
        info = header[dial]
        if dial not in wr.t.keys():
            print(f"  [skip] {dial}: not in SystWeights")
            continue
        r_cv = wr.at_cv(dial)
        if r_cv is None:
            print(f"  [skip] {dial}: cv {info['cv']:g} not in variations")
            continue
        w_ref = dial_weight(base, r_cv)
        shifts = []
        for s in sigmas:
            for sg in signs:
                r = wr.at_value(dial, info["cv"] + sg * s)
                if r is None:
                    print(f"  [info] {short(dial, args.short_labels)}: "
                          f"{sg * s:+g}σ not available")
                    continue
                shifts.append((sg * s, dial_weight(base, r)))
        print(f"  {dial}: {len(shifts)} shifted curve(s)")

        for cont, sample in blocks:
            sel = keep & cmask[cont] & np.isin(v["channel"], CHANNEL_SAMPLES[sample])
            tag = sample if cont == "all" else f"{cont}_{sample}"
            for var in args.variables:
                edges, special = channel_bins(var, sample, cont, bins, preset,
                                           explicit)
                xv = v[var][sel]
                ok = np.isfinite(xv)
                xs = xv[ok]
                if (cont, sample, var) not in reported:
                    reported.add((cont, sample, var))
                    under = np.sum(xs < edges[0])
                    over = np.sum(xs > edges[-1])
                    print(f"  {tag}/{var}: {ok.sum()} finite of {sel.sum()}; "
                          f"underflow {under}, overflow {over}; edges "
                          f"{np.round(edges, 4).tolist()}")
                cv_h = hist(xs, edges, w_ref[sel][ok])
                cv_e2 = hist(xs, edges, w_ref[sel][ok] ** 2)
                curves = []
                for sig, w in shifts:
                    k = int(round(abs(sig)))
                    curves.append({"h": hist(xs, edges, w[sel][ok]),
                                   "color": SIGMA_COLORS.get(k, "tab:gray"),
                                   "ls": "-" if sig > 0 else "--",
                                   "label": f"{sig:+g}$\\sigma$"})
                if args.show_nominal:
                    curves.append({"h": hist(xs, edges, base[sel][ok]),
                                   "color": "tab:red", "ls": ":",
                                   "label": "nominal (no dial)"})
                curves.sort(key=lambda c: c["label"])
                title = (args.title or "{dial}\n{sample}").format(
                    dial=short(dial, args.short_labels), var=var,
                    sample=CONT_LABEL[cont] + SAMPLE_LABEL[sample]
                   )
                out = os.path.join(args.outdir, tag, var,
                                   f"{dial}.{args.format}")
                draw(curves, cv_h, cv_e2, edges, args, VARIABLES[var][0],
                     title, out, logx=var in args.logx_vars,
                     uniform=args.uniform_axis and (var in custom or special))
                n_plots += 1
        wr.cache.pop(dial, None)
    print(f"Saved {n_plots} plot(s) under {args.outdir}/<[fc_|pc_]sample>/<variable>/")


if __name__ == "__main__":
    main()
