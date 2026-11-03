#!/usr/bin/env python3
"""
syst_lib.py: shared helpers for plot_syst_spectra.py and syst_covariance.py.

Not run directly. Provides:
* reading the CAF (load_caf) and the dial weight file (read_header,
  WeightReader: systsHeader name / cv / variations, per-event SystWeights);
* dial selection (match_dials, dial groups from plot_cv_weights.py, short);
* the variable catalogue, binning presets and the FC/PC binning table
  (parse_bins, sample_bins, FCPC_EDGES);
* the part_info.npz reader (load_part_info, from make_part_info.py);
* NaN-safe histogramming (hist) and the spectrum + ratio plot (draw);
* the default input files (DEFAULT_CAF, DEFAULT_WEIGHTS).

Requirements: uproot, awkward, numpy, matplotlib
"""

import fnmatch
import os
import re
import sys

import numpy as np

try:
    import uproot
    import awkward as ak
except ImportError:
    sys.exit("This script needs uproot and awkward: pip install uproot awkward")


# Dial groups: reuse the ones from plot_cv_weights.py when it sits alongside.
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from plot_cv_weights import GROUPS  # noqa: E402
except Exception:
    GROUPS = {}

# --------------------------------------------------------------------------
# Constants / variable catalogue
# --------------------------------------------------------------------------
NUE, NUMU, NC = 12, 14, 0
# sample name -> reco flavours (from the CVN selection) it contains
SAMPLES = {"nue": (NUE,), "numu": (NUMU,), "nc": (NC,),
           "nue+numu": (NUE, NUMU), "all": (NUE, NUMU, NC)}
SAMPLE_LABEL = {"nue": r"$\nu_e$ sample", "numu": r"$\nu_\mu$ sample",
                "nc": "NC sample",
                "nue+numu": r"$\nu_e + \nu_\mu$ samples",
                "all": r"$\nu_e + \nu_\mu$ + NC samples"}


# true interaction type (mc.nu.iscc)


def sample_mask(sample, reco_pdg):
    """True for events whose reco flavour belongs to `sample`."""
    return np.isin(reco_pdg, SAMPLES[sample])

# name -> (axis label, default nbins, lo, hi)
VARIABLES = {
    "reco_E":    (r"Reconstructed $E_\nu$ [GeV]",         40, 0.0, 10.0),
    "reco_Elep": (r"Reconstructed $E_{\rm lep}$ [GeV]",   40, 0.0, 10.0),
    "reco_Ehad": (r"Reconstructed $E_{\rm had}$ [GeV]",   40, 0.0, 5.0),
    "reco_cosz": (r"Reconstructed $\cos\theta_z$",        20, -1.0, 1.0),
    "true_E":    (r"True $E_\nu$ [GeV]",                  40, 0.0, 10.0),
    "true_Elep": (r"True $E_{\rm lep} = E_\nu(1-y)$ [GeV]", 40, 0.0, 10.0),
    "true_Ehad": (r"True $E_{\rm had} = E_\nu y$ [GeV]",  40, 0.0, 5.0),
    "true_cosz": (r"True $\cos\theta_z$",                 20, -1.0, 1.0),
    "reco_y":    (r"Reconstructed $y = E_{\rm had}/E_\nu$", 20, 0.0, 1.0),
    "true_y":    (r"True $y$ (inelasticity)",              20, 0.0, 1.0),
}

MC = "rec/mc/mc.nu/mc.nu."
PAN = "rec/common/common.ixn.pandora/common.ixn.pandora."
NPFPS = "rec/fd/fd.hd.pandora/fd.hd.pandora.npfps"


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------
def first_or_nan(arr, name, warn):
    """Per-entry first element of a jagged branch; NaN where empty."""
    if arr.ndim == 1:
        return ak.to_numpy(arr).astype(float)
    n = ak.to_numpy(ak.num(arr, axis=1))
    if warn and (n > 1).any():
        print(f"  [warn] {name}: {(n > 1).sum()} entries have >1 element; "
              f"using the first")
    out = ak.fill_none(ak.firsts(arr, axis=1), np.nan)
    return ak.to_numpy(out).astype(float)


HIT_COLLECTION = 3   # caf::RecoObjType::kHitCollection (duneanaobj SREnums.h)


def count_particles(t, all_keys, max_events):
    """Reco particles of the first pandora interaction, excluding the
    unassociated-hits object (origRecoObjType == kHitCollection).

    Returns a float array (NaN where there is no interaction), or None if
    the particle-type branch cannot be read.
    """
    leaf = "part.pandora.origRecoObjType"
    keys = [k for k in all_keys if k.split("/")[-1].endswith(leaf)]
    try:
        if keys:
            typ = t[min(keys, key=len)].array(library="ak",
                                              entry_stop=max_events)
        else:                       # unsplit object: read it, take the field
            obj = [k for k in all_keys
                   if k.split("/")[-1].endswith("ixn.pandora.part.pandora")]
            if not obj:
                raise KeyError("no part.pandora branch")
            typ = t[min(obj, key=len)].array(library="ak",
                                             entry_stop=max_events)
            typ = typ["origRecoObjType"]
        first = ak.firsts(typ, axis=1)            # first interaction
        n = ak.sum(first != HIT_COLLECTION, axis=-1)
        return ak.to_numpy(ak.fill_none(n, np.nan)).astype(float)
    except Exception as e:           # noqa: BLE001
        print(f"  [warn] cannot read pandora particle types: {e}")
        return None


def load_caf(path, max_events=None, part_info=None):
    branches = {
        "pdg": MC + "pdg", "E": MC + "E", "momy": MC + "momentum.y",
        "y": MC + "inelasticity", "iscc": MC + "iscc", "mode": MC + "mode",
        "lep_calo": PAN + "Enu.lep_calo", "e_calo": PAN + "Enu.e_calo",
        "mu_had": PAN + "Enu.mu_had", "e_had": PAN + "Enu.e_had",
        "lngtrk_y": PAN + "dir.lngtrk.y", "heshw_y": PAN + "dir.heshw.y",
        "calo_y": PAN + "dir.calo.y",
        "lngtrk_x": PAN + "dir.lngtrk.x", "lngtrk_z": PAN + "dir.lngtrk.z",
        "heshw_x": PAN + "dir.heshw.x", "heshw_z": PAN + "dir.heshw.z",
        "calo_x": PAN + "dir.calo.x", "calo_z": PAN + "dir.calo.z",
        "enu_calo": PAN + "Enu.calo",
        "cvn_nue": PAN + "nuhyp.cvn.nue", "cvn_numu": PAN + "nuhyp.cvn.numu",
        "cvn_nc": PAN + "nuhyp.cvn.nc", "npfps": NPFPS,
    }
    with uproot.open(path) as f:
        t = f["cafTree"]
        # Resolve each branch by its leaf name, so both
        # 'rec/mc/mc.nu/mc.nu.E' and 'rec/mc/mc.nu.E' layouts work.
        all_keys = t.keys(recursive=True, full_paths=True)
        by_leaf = {}
        for k in all_keys:
            by_leaf.setdefault(k.split("/")[-1], k)
        resolved, missing = {}, []
        for name, b in branches.items():
            leaf = b.split("/")[-1]
            if b in all_keys:
                resolved[name] = b
            elif leaf in by_leaf:
                resolved[name] = by_leaf[leaf]
            else:
                missing.append(b)
        if missing:
            sys.exit("Missing cafTree branches:\n  " + "\n  ".join(missing))
        branches = resolved
        raw = t.arrays(list(branches.values()), library="ak",
                       entry_stop=max_events)
        n_reco = ak.to_numpy(ak.num(raw[branches["cvn_nue"]], axis=1))
        d = {k: first_or_nan(raw[b], k, warn=k in ("E", "cvn_nue"))
             for k, b in branches.items()}
        d["has_reco"] = n_reco > 0
        d["n_reco"] = n_reco
        d["n_particles"] = count_particles(t, all_keys, max_events)
        # number of pandora particles per interaction (includes the
        # unassociated-hits object); used when the types cannot be read
        npan = [k for k in all_keys
                if k.split("/")[-1].endswith("ixn.pandora.part.npandora")]
        d["npandora"] = (first_or_nan(t[min(npan, key=len)].array(
            library="ak", entry_stop=max_events), "npandora", False)
            if npan else None)
        d["walldist_min"] = None
        if part_info and os.path.exists(part_info):
            d["walldist_min"], npart = load_part_info(part_info,
                                                      t.num_entries,
                                                      max_events)
            d["n_particles"] = npart          # exact type-based count
        elif part_info:
            print(f"  [info] {part_info} not found: no FC/PC split, and "
                  f"nParticles from the fallback (run make_part_info.py)")

        w = f["weights"].arrays(library="np", entry_stop=max_events)
    d.update({f"w_{k}": np.asarray(v, float) for k, v in w.items()})
    return d


def read_header(path):
    with uproot.open(path) as f:
        arr = f["systsHeader"].arrays(library="ak")
    names = [n.decode() if isinstance(n, bytes) else str(n)
             for n in ak.to_list(arr["name"])]
    cvs = ak.to_list(arr["cv"]) if "cv" in arr.fields else [0.0] * len(names)
    var = ak.to_list(arr["variations"])
    corr = (ak.to_list(arr["isCorrection"]) if "isCorrection" in arr.fields
            else [False] * len(names))
    return {n: {"cv": float(c if c is not None else 0.0),
                "variations": [float(v) for v in (vv or [])],
                "isCorrection": bool(ic)}
            for n, c, vv, ic in zip(names, cvs, var, corr)}


def value_index(info, value, tol=1e-6):
    for i, v in enumerate(info["variations"]):
        if abs(v - value) < tol:
            return i
    if len(info["variations"]) <= 1 and abs(value - info["cv"]) < tol:
        return 0          # correction / single-valued dial
    return None


class WeightReader:
    """Reads dial weights lazily from SystWeights, with a small cache."""

    def __init__(self, path, header, n_events, raw=True):
        self.f = uproot.open(path)
        self.t = self.f["SystWeights"]
        self.header = header
        self.n = n_events
        self.cache = {}
        # raw=True : stored responses as they are (negative kept, NaN/inf
        #            -> NaN so the bins they touch become missing)
        # raw=False: old behaviour (negative -> 0, NaN/inf -> 1)
        self.raw_responses = raw
        if self.t.num_entries < n_events:
            sys.exit(f"SystWeights has {self.t.num_entries} entries but the "
                     f"CAF has {n_events}: files do not match.")

    def raw(self, dial):
        if dial not in self.cache:
            self.cache[dial] = self.t[dial].array(library="ak",
                                                  entry_stop=self.n)
        return self.cache[dial]

    def at_value(self, dial, value):
        """Per-event weight of `dial` at variation `value`.

        Raw stored response by default; see __init__ for --clip-responses.
        An event with no entry at this index gets a neutral weight of 1.
        """
        idx = value_index(self.header[dial], value)
        if idx is None:
            return None
        w = self.raw(dial)
        if w.ndim == 1:
            out = ak.to_numpy(w).astype(float)
        else:
            n = ak.to_numpy(ak.num(w, axis=1))
            out = np.ones(len(w))            # missing entry -> neutral weight
            ok = n > idx
            out[ok] = ak.to_numpy(w[ok][:, idx]).astype(float)
        if self.raw_responses:
            return np.where(np.isfinite(out), out, np.nan)
        out = np.where(np.isfinite(out), out, 1.0)
        return np.clip(out, 0.0, None)

    def at_cv(self, dial):
        return self.at_value(dial, self.header[dial]["cv"])


# --------------------------------------------------------------------------
# Dial and bin selection helpers
# --------------------------------------------------------------------------
def match_dials(names, patterns, regexes=(), excludes=()):
    out = []
    for n in names:
        if (any(fnmatch.fnmatchcase(n, p) for p in patterns)
                or any(re.search(r, n) for r in regexes)) \
                and not any(fnmatch.fnmatchcase(n, e) for e in excludes):
            out.append(n)
    return out


# Shorthands accepted in --edges / presets: apply to both reco and true.
VAR_ALIASES = {
    "E":    ["reco_E", "true_E"],
    "y":    ["reco_y", "true_y"],
    "Elep": ["reco_Elep", "true_Elep"],
    "Ehad": ["reco_Ehad", "true_Ehad"],
    "cosz": ["reco_cosz", "true_cosz"],
}

# Named binning presets (variable-width edges).
BINNING_PRESETS = {
    "atm": {
        "Elep": [0, 0.28, 0.37, 0.50, 0.68, 0.92, 1.3, 1.9, 3.2, 100],
        "Ehad": [0, 0.03, 0.05, 0.09, 0.13, 0.20, 0.31, 0.53, 1.00, 2.5, 100],
        "cosz": [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1],
    },
}
# atm_fcpc = atm, plus sample-specific E_nu / y / cosZ edges for the FC and PC
# nue / numu samples (FCPC_EDGES). Other cases (containment "all", combined
# samples, NC) fall back to the atm / default edges.
BINNING_PRESETS["atm_fcpc"] = dict(BINNING_PRESETS["atm"])
_COSZ10 = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
FCPC_EDGES = {
    ("pc", "nue"): {
        "cosz": _COSZ10,
        "E": [0, 0.2, 0.4, 0.6, 0.8, 1.05, 1.4, 2, 3.25, 100],
        "y": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1]},
    ("fc", "numu"): {
        "cosz": _COSZ10,
        "E": [0, 0.4, 0.6, 0.8, 1, 1.1, 1.2, 1.45, 1.65, 1.9, 100],
        "y": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1]},
    ("fc", "nue"): {
        "cosz": _COSZ10,
        "E": [0, 0.2, 0.4, 0.6, 0.8, 1, 1.1, 1.2, 1.35, 1.5, 1.7, 1.95, 2.25,
              3.25, 100],
        "y": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1]},
    ("pc", "numu"): {
        "cosz": _COSZ10,
        "E": [0, 0.4, 0.6, 0.8, 1, 1.15, 1.3, 1.45, 1.6, 1.8, 2, 2.2, 2.45,
              2.7, 3, 3.35, 3.75, 4.25, 4.85, 6.85, 100],
        "y": [0, 0.1, 0.2, 0.3, 1]},
}
_ALIAS_OF = {v: k for k, vs in
             {"E": ["reco_E", "true_E"], "y": ["reco_y", "true_y"],
              "cosz": ["reco_cosz", "true_cosz"]}.items() for v in vs}


def explicit_bin_vars(specs, edge_specs):
    """Variables whose binning the user set with --bins / --edges."""
    out = set()
    for s in list(specs or []) + list(edge_specs or []):
        name = s.split("=", 1)[0]
        if name in VARIABLES or name in VAR_ALIASES:
            out.update(expand_var(name))
    return out


def sample_bins(var, sample, cont, bins, preset, explicit):
    """Edges for (variable, sample, containment): --bins/--edges win, then
    the FC/PC table of atm_fcpc, then the variable's general edges.
    Returns (edges, is_variable_width_preset)."""
    if var not in explicit and preset == "atm_fcpc":
        tab = FCPC_EDGES.get((cont, sample), {})
        alias = _ALIAS_OF.get(var)
        if alias in tab:
            return np.asarray(tab[alias], float), True
    return bins[var], False


# ---------------------------------------------------------------------------
# Containment (FC / PC) from the closest wall distance of the first
# interaction's particles (make_part_info.py -> walldist_min)
# ---------------------------------------------------------------------------
CONT_LABEL = {"all": "", "fc": "FC ", "pc": "PC "}


def load_part_info(path, n_tree, max_events):
    """Read make_part_info.py output; returns (walldist_min, n_particles)."""
    z = np.load(path)
    n_file = int(z["n_entries"]) if "n_entries" in z.files else len(z["walldist_min"])
    if n_file != n_tree:
        sys.exit(f"{path} was made for a cafTree with {n_file} entries, but "
                 f"this CAF has {n_tree}: rerun make_part_info.py")
    stop = max_events if max_events else len(z["walldist_min"])
    print(f"  part info from {path}: walldist_min and exact nParticles")
    return (np.asarray(z["walldist_min"][:stop], float),
            np.asarray(z["n_particles"][:stop], float))


def expand_var(name):
    if name in VARIABLES:
        return [name]
    if name in VAR_ALIASES:
        return VAR_ALIASES[name]
    sys.exit(f"Unknown variable '{name}'. Use one of "
             f"{sorted(VARIABLES) + sorted(VAR_ALIASES)}")


def set_edges(bins, custom, name, edges):
    edges = np.asarray(edges, float)
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
        sys.exit(f"Edges for '{name}' must be >= 2 strictly increasing "
                 f"numbers, got {edges.tolist()}")
    for v in expand_var(name):
        bins[v] = edges
        custom.add(v)


def parse_bins(specs, nbins_override, edge_specs=(), preset=None):
    """Return ({var: edges}, set of vars with custom variable-width edges).

    Priority (later wins): defaults -> --nbins -> --binning preset
                           -> --bins var=n,lo,hi -> --edges var=e0,e1,...
    """
    bins, custom = {}, set()
    for name, (_, nb, lo, hi) in VARIABLES.items():
        bins[name] = np.linspace(lo, hi, (nbins_override or nb) + 1)
    if preset:
        for name, edges in BINNING_PRESETS[preset].items():
            set_edges(bins, custom, name, edges)
    for s in specs or []:
        try:
            name, rest = s.split("=")
            nb, lo, hi = rest.split(",")
            for v in expand_var(name):
                bins[v] = np.linspace(float(lo), float(hi), int(nb) + 1)
                custom.discard(v)
        except ValueError:
            sys.exit(f"Bad --bins spec '{s}', expected var=nbins,lo,hi")
    for s in edge_specs or []:
        try:
            name, rest = s.split("=", 1)
            rest = rest.strip().strip("[]()")
            # accept both ASCII '-' and the unicode minus sign
            edges = [float(x.replace("\u2212", "-"))
                     for x in rest.replace(";", ",").split(",") if x.strip()]
        except ValueError:
            sys.exit(f"Bad --edges spec '{s}', expected var=e0,e1,e2,...")
        set_edges(bins, custom, name, edges)
    return bins, custom


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
SIGMA_COLORS = {1: "tab:blue", 2: "tab:orange", 3: "tab:green",
                4: "tab:purple", 5: "tab:brown"}


def hist(x, bins, w):
    """Weighted histogram with np.histogram's bin convention ([lo, hi),
    last bin closed). Built with bincount so that a nonfinite weight makes
    only the bin it falls in missing (np.histogram would spread the NaN to
    the following bins through its cumulative sum)."""
    bins = np.asarray(bins, float)
    x = np.asarray(x, float)
    nb = len(bins) - 1
    idx = np.searchsorted(bins, x, side="right") - 1
    idx[x == bins[-1]] = nb - 1
    ok = np.isfinite(x) & (idx >= 0) & (idx < nb)
    return np.bincount(idx[ok], weights=np.asarray(w, float)[ok],
                       minlength=nb)


def draw(curves, cv, cv_err2, bins, args, xlabel, title, outpath, logx,
         uniform=False):
    import matplotlib.pyplot as plt

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=args.figsize, sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})

    true_edges = np.asarray(bins, float)
    if args.per_width:                       # events per unit of x
        width = np.diff(true_edges)
        cv = cv / width
        cv_err2 = cv_err2 / width ** 2
        curves = [dict(c, h=c["h"] / width) for c in curves]
    if uniform:                              # equal-width bins on the page
        bins = np.arange(len(true_edges), dtype=float)
        logx = False

    def step(a, y, **kw):
        a.stairs(y, bins, **kw)

    step(ax, cv, color="black", lw=2, label="CV")
    if args.stat_band:
        err = np.sqrt(cv_err2)
        ax.stairs(cv + err, bins, baseline=cv - err, fill=True,
                  color="gray", alpha=0.3, label="CV stat. unc.")
    for c in curves:
        step(ax, c["h"], color=c["color"], ls=c["ls"], lw=args.linewidth,
             label=c["label"])

    with np.errstate(divide="ignore", invalid="ignore"):
        axr.axhline(1.0, color="black", lw=1)
        for c in curves:
            r = np.where(cv > 0, c["h"] / cv, np.nan)
            # baseline=None: no vertical drops to 0 at the edges / empty bins
            axr.stairs(r, bins, baseline=None, color=c["color"], ls=c["ls"],
                       lw=args.linewidth)
        if args.stat_band:
            rel = np.where(cv > 0, np.sqrt(cv_err2) / cv, np.nan)
            axr.stairs(1 + rel, bins, baseline=1 - rel, fill=True,
                       color="gray", alpha=0.3)

    unit = ""
    if args.per_width:
        unit = " / GeV" if "GeV" in xlabel else " / unit"
    ax.set_ylabel(args.ylabel or f"Events{unit}", fontsize=args.label_fontsize)
    if uniform:
        axr.set_xticks(bins)
        axr.set_xticklabels([f"{e:g}" for e in true_edges],
                            rotation=45 if len(true_edges) > 8 else 0,
                            fontsize=args.label_fontsize * 0.8)
        axr.set_xlim(bins[0], bins[-1])
    ax.set_title(title, fontsize=args.title_fontsize)
    axr.set_xlabel(xlabel, fontsize=args.label_fontsize)
    axr.set_ylabel("Ratio to CV", fontsize=args.label_fontsize * 0.85)
    rs = [np.where(cv > 0, c["h"] / np.where(cv > 0, cv, 1), np.nan)
          for c in curves]
    vals = np.concatenate([r[np.isfinite(r)] for r in rs]) if rs else np.array([])
    if args.ratio_logy:
        from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
        axr.set_yscale("log")
        nice = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
        axr.yaxis.set_major_locator(FixedLocator(nice))
        axr.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        axr.yaxis.set_minor_formatter(NullFormatter())
    if args.ratio_range:
        axr.set_ylim(*args.ratio_range)
    elif len(vals):
        if args.ratio_logy:
            pos = vals[vals > 0]
            lo = min(pos.min(), 1.0) if len(pos) else 0.5
            hi = max(vals.max(), 1.0)
            axr.set_ylim(lo / 1.3, hi * 1.3)
        else:
            # fit all curves; asymmetric around 1, padded, never below 0
            lo = min(vals.min(), 1.0)
            hi = max(vals.max(), 1.0)
            span = max(hi - lo, 0.04)
            axr.set_ylim(max(lo - 0.12 * span, 0.0), hi + 0.12 * span)
    if args.logy:
        ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
        pos = true_edges[true_edges > 0]
        if len(pos):
            ax.set_xlim(left=pos[0] if true_edges[0] <= 0 else true_edges[0])
    if args.grid_lines:
        ax.grid(alpha=0.3)
        axr.grid(alpha=0.3)
    if args.legend:
        ax.legend(loc=args.legend_loc, fontsize=args.legend_fontsize,
                  ncol=args.legend_ncol, frameon=True)
    fig.align_ylabels([ax, axr])
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    fig.savefig(outpath, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------
# Default input files (override with --caf / --weights)
DEFAULT_CAF = "/exp/dune/data/users/pgranger/caf_new_sum.2.6M_weighted.root"
DEFAULT_WEIGHTS = "/exp/dune/data/users/tthuynh/merged_new_systs_wsbn_hotfixed_full_header.root"

PREFIXES = ("GENIEReWeight_ICARUS_v2_multisigma_", "CCQETemplateReweight_SBN_v3_",
            "MECq0q3InterpWeighting_SBN_v3_", "MiscInteractionSysts_DUNE_TDR_",
            "ZExpPCAWeighter_SBN_v3_", "QEInterference_SBN_v3_",
            "PionAbsWeighter_SBN_v3_", "CCQEXSecCorr_SBN_v3_",
            "ResIso_ResIso_", "WSReweight_v1_")


def short(name, on):
    if on:
        for pre in PREFIXES:
            if name.startswith(pre):
                return name[len(pre):]
    return name

