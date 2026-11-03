#!/usr/bin/env python3
"""
plot_syst_spectra.py

Spectra of reco and true variables for atmospheric-neutrino CAFs, showing the
effect of cross-section dials at their central value (CV) and at +/- n sigma.

Inputs
------
* CAF file     : cafTree (rec/mc/mc.nu, rec/common/common.ixn.pandora,
                 rec/fd/fd.hd.pandora.npfps) + 'weights' tree (newest cycle).
* Weight file  : SystWeights (per-event vector per dial) + systsHeader
                 (name, cv, variations, id, isCorrection).
Both files are matched entry-by-entry (same number of events, same order).

Definitions follow pgranger23/systematics-analysis systs.py
-----------------------------------------------------------
reco flavour : CVN thresholds (--selection; see select_flavour) -> nue / numu / nc
reco E_nu    : numu -> Enu.lep_calo        ; else -> Enu.e_calo
reco E_had   : numu -> Enu.mu_had          ; else -> Enu.e_had
reco E_lep   : reco E_nu - reco E_had
reco cosZ    : -d.y / |d|  (negative = upgoing), default --cosz mach3:
               nParticles = pandora particles of the first interaction,
                            excluding origRecoObjType == kHitCollection (3)
               d = dir.calo   if nParticles < 2 or !(Enu.calo < 1.3 GeV)
                   dir.heshw  else, for the nue-like sample
                   dir.lngtrk else (numu-like and NC samples)
               (--cosz calo / systs / detsys give the older estimators)
true cosZ    : -p_y / E of the neutrino
true E_lep   : E * (1 - y)        true E_had : E * y      (y = inelasticity)
quality cut  : npfps >= 2, reco interaction present, reco direction != 0

Event weight
------------
base      = final_oscillated_w   (CAF 'weights' tree, oscillated)
CV        = base * prod_{d in CV dials} w_d(cv)                   (negative w -> 0)
dial X@v  = base * prod_{d != X} w_d(cv) * w_X(v)
X off     = base * prod_{d != X} w_d(cv)                          (--show-off)

Requirements: uproot, awkward, numpy, matplotlib

Examples
--------
python plot_syst_spectra.py --caf caf.root --weights syst.root --list
python plot_syst_spectra.py --caf caf.root --weights syst.root \
    --group FSI_Pion --sigmas 1 2 3 --outdir plots_fsi
python plot_syst_spectra.py --caf caf.root --weights syst.root \
    --dials "*MaCCRES" --variables reco_Elep true_Elep --samples numu \
    --bins reco_Elep=40,0,10 --logy --show-off
"""

import argparse
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

import matplotlib

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
INTERACTIONS = ("all", "cc", "nc")
INTERACTION_LABEL = {"all": "", "cc": ", true CC", "nc": ", true NC"}


def interaction_mask(interaction, iscc):
    """all: CC + NC; cc: mc.nu.iscc true; nc: mc.nu.iscc false."""
    if interaction == "cc":
        return iscc > 0
    if interaction == "nc":
        return iscc == 0           # NaN (no true neutrino) is neither
    return np.ones(len(iscc), dtype=bool)


def resolve_interaction(args):
    """--cc-only is kept as a shortcut for --interaction cc."""
    if getattr(args, "cc_only", False):
        if args.interaction not in ("all", "cc"):
            sys.exit("--cc-only conflicts with --interaction "
                     f"{args.interaction}")
        args.interaction = "cc"


def check_containment_args(args, d):
    """FC / PC need walldist from make_part_info.py."""
    if any(c != "all" for c in args.containment) and d.get("walldist_min") is None:
        sys.exit("--containment fc / pc needs the particle wall distances: run "
                 "make_part_info.py (with ROOT) to make "
                 f"{args.part_info}, or pass --part-info <file>")
    if d.get("walldist_min") is not None:
        wd = d["walldist_min"]
        print(f"  containment (cut {args.walldist_cut:g} cm): FC {np.sum(wd > args.walldist_cut)}, "
              f"PC {np.sum(wd <= args.walldist_cut)}, undefined {np.sum(~np.isfinite(wd))}")


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
DEFAULT_VARS = ["reco_Elep", "reco_Ehad", "reco_cosz",
                "true_Elep", "true_Ehad", "true_cosz"]

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


# Dials named <provider>_b<N> (e.g. ZExpPCAWeighter_SBN_v3_MvA_b1..b4) are
# scans of one provider that share the same zero-displacement model: their CV
# response is the same correction, so it is applied once, not once per scan.
SHARED_CV_RE = re.compile(r"^(.*)_b\d+$")


def shared_cv_keys(names, independent=False):
    """dial -> CV-group key; dials of one b-scan provider share a key."""
    key = {nm: nm for nm in names}
    if independent:
        return key
    groups = {}
    for nm in names:
        m = SHARED_CV_RE.match(nm)
        if m:
            groups.setdefault(m.group(1), []).append(nm)
    for prefix, members in groups.items():
        if len(members) > 1:
            for nm in members:
                key[nm] = prefix + "_b*"
    return key


class CVWeight:
    """Product of the dials' CV responses, with each shared b-scan provider
    counted once. others(dial) = the product without that dial's group."""

    def __init__(self, wr, header, cv_dials, keep_cached=(),
                 independent_b=False):
        self.n = wr.n
        self.key = shared_cv_keys(list(header), independent_b)
        self.w, self.members = {}, {}
        self.total = np.ones(self.n)
        for dial in cv_dials:
            w = wr.at_cv(dial)
            if w is None:
                print(f"  [warn] {dial}: cv {header[dial]['cv']:g} not in "
                      f"variations, not applied")
                continue
            k = self.key[dial]
            if k in self.w:
                if not np.allclose(w, self.w[k], equal_nan=True):
                    print(f"  [warn] {dial}: CV response differs from "
                          f"{self.members[k][0]} although both are scans of "
                          f"{k}; using {self.members[k][0]}'s once")
                self.members[k].append(dial)
            else:
                self.w[k], self.members[k] = w, [dial]
                self.total = self.total * w
            if dial not in keep_cached:
                wr.cache.pop(dial, None)
        shared = {k: m for k, m in self.members.items() if len(m) > 1}
        print(f"  CV weight: {len(self.w)} independent factor(s) from "
              f"{sum(len(m) for m in self.members.values())} dial(s)")
        for k, m in shared.items():
            print(f"    shared zero response, applied once: {k} "
                  f"({len(m)} scans)")
        nonfin = np.sum(~np.isfinite(self.total))
        neg = np.sum(self.total < 0)
        if nonfin or neg:
            print(f"    {nonfin} events with a nonfinite CV response "
                  f"(their bins become missing), {neg} with a negative one")

    def others(self, dial):
        """CV product without the factor of `dial`'s group."""
        k = self.key.get(dial, dial)
        if k not in self.w:
            return self.total
        w = self.w[k]
        safe = np.isfinite(w) & (w != 0)
        out = np.empty(self.n)
        with np.errstate(invalid="ignore", over="ignore"):
            out[safe] = self.total[safe] / w[safe]
        if (~safe).any():          # recompute exactly where dividing fails
            prod = np.ones((~safe).sum())
            for k2, w2 in self.w.items():
                if k2 != k:
                    prod = prod * w2[~safe]
            out[~safe] = prod
        return out


# --------------------------------------------------------------------------
# Derived quantities (systs.py conventions)
# --------------------------------------------------------------------------
# CVN selection thresholds (override with --cvn-numu-cut / --cvn-nue-cut)
CVN_NUMU_CUT = 0.56
CVN_NUE_CUT = 0.55


def select_flavour(cvn_numu, cvn_nue, numu_cut=CVN_NUMU_CUT,
                   nue_cut=CVN_NUE_CUT):
    """Reco sample from CVN scores.

    pass_mu = cvn_numu > numu_cut,  pass_e = cvn_nue > nue_cut
      neither passes          -> NC
      only pass_mu            -> numu
      only pass_e             -> nue
      both pass               -> numu if cvn_numu > cvn_nue, else nue
    A score exactly at the cut does not pass; NaN scores never pass.
    """
    pass_mu = cvn_numu > numu_cut
    pass_e = cvn_nue > nue_cut
    both = pass_mu & pass_e
    return np.select(
        [both & (cvn_numu > cvn_nue), both, pass_mu, pass_e],
        [NUMU, NUE, NUMU, NUE], default=NC)


def particle_count(d, source="auto"):
    """nParticles for the MaCh3 direction (= NbPFPs - 1).

    type     : count part.pandora with origRecoObjType != kHitCollection
    npandora : part.npandora - 1 (one unassociated-hits object assumed)
    npfps    : fd.hd.pandora.npfps - 1 (npfps assumed to include the nu PFP)
    auto     : type if readable, otherwise npandora
    """
    have_type = d.get("n_particles") is not None
    if source == "auto":
        source = "type" if have_type else "npandora"
        if not have_type:
            print("  [warn] particle types unreadable (memberwise-stored "
                  "branch) and no --part-info file: using nParticles = "
                  "part.npandora - 1, i.e. assuming exactly one "
                  "unassociated-hits object per interaction (run "
                  "make_part_info.py for the exact count)")
    if source == "type":
        if not have_type:
            sys.exit("--nparticles-from type: part.pandora.origRecoObjType "
                     "could not be read; use npandora or npfps")
        n = d["n_particles"]
    elif source == "npandora":
        if d.get("npandora") is None:
            sys.exit("common.ixn.pandora.part.npandora not found; use "
                     "--nparticles-from npfps or another --cosz")
        n = d["npandora"] - 1
    else:
        n = d["npfps"] - 1
    if d.get("npandora") is not None:
        both = np.isfinite(d["npandora"]) & np.isfinite(d["npfps"])
        if both.any():
            agree = np.mean(d["npandora"][both] == d["npfps"][both])
            print(f"  nParticles from {source}; check: part.npandora == "
                  f"fd.hd.pandora.npfps for {100 * agree:.1f}% of events")
    return n


def derive(d, numu_cut=CVN_NUMU_CUT, nue_cut=CVN_NUE_CUT,
           selection="cuts", cosz="mach3", base_weight="flux_xsec",
           nparticles_from="auto"):
    nue, numu, nc = d["cvn_nue"], d["cvn_numu"], d["cvn_nc"]
    if selection == "argmax":           # original systs.py behaviour
        reco_pdg = np.where(nue > numu,
                            np.where(nue > nc, NUE, NC),
                            np.where(numu > nc, NUMU, NC))
    elif selection == "mach3":          # MaCh3 / detsys: numu wins if both pass
        reco_pdg = np.select([numu > numu_cut, nue > nue_cut],
                             [NUMU, NUE], default=NC)
    else:
        reco_pdg = select_flavour(numu, nue, numu_cut, nue_cut)
    is_numu = reco_pdg == NUMU

    reco_E = np.where(is_numu, d["lep_calo"], d["e_calo"])
    reco_Ehad = np.where(is_numu, d["mu_had"], d["e_had"])

    if cosz == "mach3":
        # MaCh3 atmospheric reco direction:
        #   nParticles = pandora particles excluding the hit collection
        #   useHits    = nParticles < 2 or !(Enu.calo < 1.3 GeV)  (NaN -> hits)
        #   direction  = dir.calo if useHits, else dir.heshw for the nue-like
        #                sample, else dir.lngtrk
        #   cosZ       = -d.y / |d|
        n_part = particle_count(d, nparticles_from)
        use_hits = (n_part < 2) | ~(d["enu_calo"] < 1.3)
        is_e = reco_pdg == NUE
        comp = {}
        for c in ("x", "y", "z"):
            comp[c] = np.where(use_hits, d["calo_" + c],
                               np.where(is_e, d["heshw_" + c],
                                        d["lngtrk_" + c]))
        norm = np.sqrt(comp["x"] ** 2 + comp["y"] ** 2 + comp["z"] ** 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            reco_cosz = -comp["y"] / norm          # |d| = 0 -> NaN
        dir_check = norm
    elif cosz == "calo":
        # calorimetric direction for every sample
        reco_cosz = -d["calo_y"]
        dir_check = reco_cosz
    elif cosz == "detsys":
        # Claire's detsys Reader.cxx: numu -> lngtrk, nue / NC -> heshw,
        # no fallback (a zero direction stays at cos(theta_z) = 0)
        reco_cosz = np.where(is_numu, -d["lngtrk_y"], -d["heshw_y"])
        dir_check = reco_cosz
    else:
        # "systs": per-sample estimator with fallback, as in systs.py
        direc_numu, direc_nue, direc_nc = (-d["lngtrk_y"], -d["heshw_y"],
                                           -d["heshw_y"])
        fallback = (d["npfps"] < 3) | (d["e_calo"] > 1.3)
        flav_dir = np.where(is_numu, direc_numu,
                            np.where(reco_pdg == NUE, direc_nue, direc_nc))
        reco_cosz = np.where(fallback | (flav_dir == 0) | (reco_pdg == NC),
                             direc_nc, flav_dir)
        dir_check = direc_nc

    E, y = d["E"], d["y"]
    v = {
        "reco_pdg": reco_pdg,
        "reco_E": reco_E,
        "reco_Ehad": reco_Ehad,
        "reco_Elep": reco_E - reco_Ehad,
        "reco_cosz": reco_cosz,
        "true_E": E,
        "true_Elep": E * (1.0 - y),
        "true_Ehad": E * y,
        "true_cosz": -d["momy"] / E,
        "reco_y": np.where(reco_E > 0, reco_Ehad / np.where(reco_E > 0,
                                                            reco_E, 1),
                           np.nan),
        "true_y": y,
    }
    # direction must be reconstructed (non-zero, finite) for the estimator used
    quality = (d["has_reco"] & (d["npfps"] >= 2) & np.isfinite(nue)
               & (dir_check != 0) & np.isfinite(dir_check))
    v["quality"] = quality

    # Base weight
    #   flux_xsec (default): unoscillated xsec x flux of the event's own true
    #     flavour: |pdg| = 12 -> flux_nue, |pdg| = 14 -> flux_numu, other -> 0
    #   final: final_oscillated_w as stored
    if base_weight == "final":
        base = d["w_final_oscillated_w"]
    else:
        apdg = np.abs(d["pdg"])
        flux = np.select([apdg == 12, apdg == 14],
                         [d["w_flux_nue"], d["w_flux_numu"]], default=0.0)
        base = d["w_xsec"] * flux
    v["base_w"] = np.nan_to_num(base, nan=0.0)
    return v


# --------------------------------------------------------------------------
# Compatibility with Claire's detsys tools
# (DUNE/atmospherics-tools, branch cdalmazz/feature/cAFComparisonTools,
#  detsys/src/Reader.cxx + SampleManager.cxx)
# --------------------------------------------------------------------------
RECO_VARS = ("reco_E", "reco_Elep", "reco_Ehad", "reco_cosz")


def apply_detsys_args(args):
    """--detsys-compatible: switch the options that differ from detsys."""
    args.cosz = "detsys"
    args.selection = "mach3"
    args.no_quality_cut = True
    args.include_nubar = True
    if hasattr(args, "per_width"):
        args.per_width = True
    if getattr(args, "ylabel", "unset") is None:
        args.ylabel = "Number of Events/Bin Width"
    print("--detsys-compatible: --cosz detsys --selection mach3 "
          "--no-quality-cut --include-nubar"
          + (" --per-width" if hasattr(args, "per_width") else "")
          + "; events without exactly one pandora interaction get no reco "
            "value (they fall outside every bin)")


def detsys_reco_mask(v, d):
    """detsys: reco values only for events with exactly one pandora ixn."""
    bad = d["n_reco"] != 1
    for k in RECO_VARS:
        v[k] = np.where(bad, np.nan, v[k])
    return int(bad.sum())


# --------------------------------------------------------------------------
# Selection helpers
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
CONTAINMENTS = ("all", "fc", "pc")
CONT_LABEL = {"all": "", "fc": "FC ", "pc": "PC "}


def containment_mask(cont, walldist_min, cut):
    """fc: walldist_min > cut; pc: walldist_min <= cut; all: every event.
    Events without a defined walldist are in neither FC nor PC."""
    if cont == "all":
        return np.ones(len(walldist_min), dtype=bool)
    if cont == "fc":
        return walldist_min > cut
    return walldist_min <= cut


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
# CLI
# --------------------------------------------------------------------------
# Default input files (override with --caf / --weights)
DEFAULT_CAF = "/exp/dune/data/users/pgranger/caf_new_sum.2.6M_weighted.root"
DEFAULT_WEIGHTS = "/exp/dune/data/users/tthuynh/merged_new_systs_wsbn_hotfixed_full_header.root"

def parse_args():
    p = argparse.ArgumentParser(
        description="Reco/true spectra with cross-section dial variations "
                    "(CV and +/- n sigma).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False)          # no silent prefix matching of options
    p.add_argument("--caf", default=DEFAULT_CAF,
                   help="CAF file (cafTree + weights)")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS,
                   help="Weight file (SystWeights + systsHeader)")
    p.add_argument("--max-events", type=int, default=None,
                   help="Only read the first N events (quick tests)")

    g = p.add_argument_group("dials")
    g.add_argument("--list", action="store_true",
                   help="List dials, cv and variations, then exit")
    g.add_argument("--group", choices=sorted(GROUPS) if GROUPS else None,
                   help="Predefined group (from plot_cv_weights.py)")
    g.add_argument("--dials", nargs="*", default=[],
                   help="Dial names or glob patterns to vary")
    g.add_argument("--regex", nargs="*", default=[])
    g.add_argument("--exclude", nargs="*", default=[])
    g.add_argument("--sigmas", nargs="*", type=float, default=[1, 2, 3],
                   help="Draw CV +/- these sigma values")
    g.add_argument("--only-up", action="store_true", help="Only +n sigma")
    g.add_argument("--only-down", action="store_true", help="Only -n sigma")
    g.add_argument("--show-off", action="store_true",
                   help="Also draw the varied dial switched off (weight 1)")
    g.add_argument("--cv-dials", nargs="*", default=["*"],
                   help="Dials held at CV in every curve (globs). "
                        "Default: all dials in systsHeader")
    g.add_argument("--no-cv", action="store_true",
                   help="Do not apply other dials' CV weights "
                        "(systs.py behaviour)")

    g = p.add_argument_group("samples & variables")
    g.add_argument("--samples", nargs="*", default=["nue", "numu", "nc"],
                   choices=list(SAMPLES),
                   help="Reco CVN samples. nue+numu adds the nue and numu "
                        "samples into one spectrum; all also adds NC")
    g.add_argument("--variables", nargs="*", default=DEFAULT_VARS,
                   choices=list(VARIABLES))
    g.add_argument("--include-nubar", action="store_true",
                   help="Keep antineutrinos (default: true neutrinos only)")
    g.add_argument("--containment", nargs="+", choices=CONTAINMENTS,
                   default=["all"],
                   help="fc: closest particle wall distance > --walldist-cut, "
                        "pc: <= cut, all: no split. Several give one set of "
                        "plots each (needs --part-info)")
    g.add_argument("--walldist-cut", type=float, default=20.0,
                   help="FC/PC boundary [cm]")
    g.add_argument("--part-info", default="part_info.npz",
                   help="Output of make_part_info.py (walldist and exact "
                        "nParticles); used if the file exists")
    g.add_argument("--interaction", choices=INTERACTIONS, default="all",
                   help="True interaction type (mc.nu.iscc): cc, nc, or "
                        "all = CC + NC")
    g.add_argument("--cc-only", action="store_true",
                   help="Shortcut for --interaction cc")
    g.add_argument("--no-quality-cut", action="store_true",
                   help="Skip npfps>=2 / reco-present / direction!=0 cut")
    g.add_argument("--base-weight", choices=["flux_xsec", "final"],
                   default="flux_xsec",
                   help="flux_xsec: unoscillated xsec x flux of the event's "
                        "true flavour (flux_nue for nue/nuebar, flux_numu "
                        "for numu/numubar); final: final_oscillated_w")
    g.add_argument("--clip-responses", action="store_true",
                   help="Old dial-weight handling: negative -> 0, "
                        "nonfinite -> 1 (default: raw stored responses; "
                        "nonfinite ones make the bins they touch missing)")
    g.add_argument("--independent-b-scans", action="store_true",
                   help="Multiply the CV response of every <provider>_bN "
                        "scan (default: once per provider)")
    g.add_argument("--nparticles-from", choices=["auto", "type", "npandora",
                                                  "npfps"], default="auto",
                   help="nParticles for --cosz mach3: count of non-hit-"
                        "collection particles (type), part.npandora - 1, or "
                        "fd.hd.pandora.npfps - 1; auto = type if readable, "
                        "else npandora")
    g.add_argument("--cosz", choices=["mach3", "calo", "systs", "detsys"],
                   default="mach3",
                   help="reco cos(theta_z). mach3 (default): -d.y/|d| with "
                        "d = dir.calo if <2 pandora particles (hit "
                        "collection excluded) or Enu.calo >= 1.3 GeV / NaN, "
                        "else dir.heshw (nue-like) or dir.lngtrk; calo: "
                        "-dir.calo.y; systs: systs.py per-sample estimator; "
                        "detsys: numu->lngtrk, nue/NC->heshw, no fallback")
    g.add_argument("--selection", choices=["cuts", "argmax", "mach3"],
                   default="cuts",
                   help="Sample selection: CVN thresholds with the larger "
                        "score winning when both pass (cuts), the same "
                        "thresholds with numu winning (mach3, as in MaCh3 "
                        "and detsys), or the systs.py CVN argmax")
    g.add_argument("--detsys-compatible", action="store_true",
                   help="Match Claire's detsys plots: --cosz detsys, "
                        "--selection mach3, --no-quality-cut, "
                        "--include-nubar, --per-width, y label 'Number of "
                        "Events/Bin Width', and reco values only for events "
                        "with exactly one pandora interaction")
    g.add_argument("--cvn-numu-cut", type=float, default=CVN_NUMU_CUT,
                   help="numu-like if cvn_numu > this")
    g.add_argument("--cvn-nue-cut", type=float, default=CVN_NUE_CUT,
                   help="nue-like if cvn_nue > this")

    g = p.add_argument_group("binning")
    g.add_argument("--nbins", type=int, default=None,
                   help="Override the bin count for every variable")
    g.add_argument("--bins", nargs="*", default=[],
                   help="Per-variable binning, e.g. reco_Elep=40,0,10")
    g.add_argument("--edges", nargs="*", default=[],
                   help="Custom variable-width bin edges, e.g. "
                        "Elep=0,0.28,0.37,0.5,100  cosz=-1,-0.5,0,0.5,1  "
                        "true_Ehad=0,0.1,1,5. Shorthands E/Elep/Ehad/cosz "
                        "set both reco_ and true_")
    g.add_argument("--binning", choices=sorted(BINNING_PRESETS) + ["none"],
                   default="atm_fcpc",
                   help="Binning preset. atm: DUNE atmospheric Elep / Ehad / "
                        "cosZ edges; atm_fcpc (default): atm plus the FC/PC "
                        "nue/numu E_nu, y and cosZ edges; none: uniform")
    g.add_argument("--uniform-axis", action="store_true",
                   help="Draw every bin with equal width on the page and "
                        "label the true edges (good for very wide last bins). "
                        "Applies to variables with custom edges")
    g.add_argument("--per-width", action="store_true",
                   help="Divide each bin by its width (events / GeV)")
    g.add_argument("--logx", action="store_true",
                   help="Log x axis for every plotted variable that allows "
                        "it (energies); cos(theta_z) stays linear")
    g.add_argument("--logx-vars", nargs="*", default=[],
                   help="Variables to draw with log x (uniform bins become "
                        "log-spaced; custom edges are kept)")

    g = p.add_argument_group("style")
    g.add_argument("--title", default=None,
                   help="Title template; {dial} {sample} {var} are filled in")
    g.add_argument("--ylabel", default=None)
    g.add_argument("--title-fontsize", type=float, default=13)
    g.add_argument("--label-fontsize", type=float, default=12)
    g.add_argument("--logy", action="store_true")
    g.add_argument("--linewidth", type=float, default=1.5)
    g.add_argument("--stat-band", action="store_true",
                   help="Shade the CV statistical uncertainty")
    g.add_argument("--ratio-range", nargs=2, type=float, default=None,
                   metavar=("YMIN", "YMAX"),
                   help="Fix the ratio-panel y range (default: fit all curves)")
    g.add_argument("--ratio-logy", action="store_true",
                   help="Log y axis in the ratio panel (for very large shifts)")
    g.add_argument("--no-legend", dest="legend", action="store_false")
    g.add_argument("--legend-loc", default="best")
    g.add_argument("--legend-fontsize", type=float, default=9)
    g.add_argument("--legend-ncol", type=int, default=1)
    g.add_argument("--short-labels", action="store_true",
                   help="Strip the generator prefix from dial names")
    g.add_argument("--grid-lines", action="store_true")
    g.add_argument("--figsize", nargs=2, type=float, default=(8, 7),
                   metavar=("W", "H"))

    g = p.add_argument_group("output")
    g.add_argument("--outdir", default="syst_spectra")
    g.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    g.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


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


def main():
    args = parse_args()
    matplotlib.use("Agg")

    header = read_header(args.weights)
    if args.list:
        print(f"{'dial':70s} {'cv':>5s}  variations")
        for n, i in header.items():
            print(f"{n:70s} {i['cv']:>5g}  {i['variations']}")
        return

    patterns = list(args.dials) + (GROUPS.get(args.group, [])
                                   if args.group else [])
    if not patterns and not args.regex:
        sys.exit("Choose dials with --group, --dials or --regex (or --list).")
    vary = match_dials(header, patterns, args.regex, args.exclude)
    if not vary:
        sys.exit("No dials matched.")
    cv_dials = [] if args.no_cv else match_dials(header, args.cv_dials)

    # ---- data -------------------------------------------------------------
    print("Reading CAF ...")
    d = load_caf(args.caf, args.max_events, args.part_info)
    check_containment_args(args, d)
    n = len(d["E"])
    if len(d["w_nue_w"]) != n:
        sys.exit(f"'weights' tree has {len(d['w_nue_w'])} entries, "
                 f"cafTree has {n}.")
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

    # ---- CV weight: product of the dials' CV responses ---------------------
    print(f"Building CV weight from {len(cv_dials)} dial(s) ...")
    cvw = CVWeight(wr, header, cv_dials, keep_cached=vary,
                   independent_b=args.independent_b_scans)
    others_at_cv = cvw.others

    # ---- event selection ---------------------------------------------------
    keep = np.isfinite(v["true_E"]) & (v["true_E"] > 0)
    if not args.include_nubar:
        keep &= d["pdg"] > 0
    keep &= interaction_mask(args.interaction, d["iscc"])
    if not args.no_quality_cut:
        keep &= v["quality"]
    print(f"Events: {n} total, {keep.sum()} after selection "
          f"(true interaction: {args.interaction})")

    preset = None if args.binning == "none" else args.binning
    bins, custom = parse_bins(args.bins, args.nbins, args.edges, preset)
    explicit = explicit_bin_vars(args.bins, args.edges)
    logx_vars = set(args.logx_vars)
    if args.logx:
        logx_vars |= set(args.variables)
    for var in sorted(logx_vars):
        if var.endswith("cosz") or bins[var][-1] <= 0:
            if var in args.variables:
                print(f"  [info] {var}: range includes negative values, "
                      f"keeping linear x")
            logx_vars.discard(var)
            continue
        if var not in custom:
            # start at the first non-zero linear edge, then log-space
            e = bins[var]
            lo = e[0] if e[0] > 0 else e[1]
            bins[var] = np.geomspace(lo, e[-1], len(e))
    args.logx_vars = sorted(logx_vars)
    for var in args.variables:
        if var in custom:
            print(f"  binning {var}: {bins[var].tolist()}")
    for cont in args.containment:
        for sample in args.samples:
            for var in args.variables:
                e, special = sample_bins(var, sample, cont, bins, preset,
                                         explicit)
                if special:
                    print(f"  binning {cont}/{sample}/{var}: {e.tolist()}")
    cmask = {c: containment_mask(c, d["walldist_min"], args.walldist_cut)
             if c != "all" else np.ones(len(keep), bool)
             for c in args.containment}

    sigmas = sorted({abs(s) for s in args.sigmas if s != 0})
    signs = [+1, -1]
    if args.only_up:
        signs = [+1]
    if args.only_down:
        signs = [-1]

    base = v["base_w"]
    n_plots = 0
    for dial in vary:
        info = header[dial]
        if dial not in wr.t.keys():
            print(f"  [skip] {dial}: not in SystWeights")
            continue
        rest = base * others_at_cv(dial)
        w_cv_dial = wr.at_cv(dial)
        if w_cv_dial is None:
            print(f"  [skip] {dial}: cv {info['cv']:g} not in variations")
            continue
        w_central = rest * w_cv_dial

        shifts = []
        for s in sigmas:
            for sg in signs:
                val = info["cv"] + sg * s
                w = wr.at_value(dial, val)
                if w is None:
                    print(f"  [info] {short(dial, args.short_labels)}: "
                          f"{sg * s:+g}σ not available")
                    continue
                shifts.append((sg * s, rest * w))
        print(f"  {dial}: {len(shifts)} shifted curve(s)")

        for cont, sample in [(c, s) for c in args.containment
                             for s in args.samples]:
            sel = keep & cmask[cont] & sample_mask(sample, v["reco_pdg"])
            tag = sample if cont == "all" else f"{cont}_{sample}"
            for var in args.variables:
                edges, special = sample_bins(var, sample, cont, bins, preset,
                                             explicit)
                x = v[var][sel]
                ok = np.isfinite(x)
                x = x[ok]
                cv_h = hist(x, edges, w_central[sel][ok])
                cv_e2 = hist(x, edges, w_central[sel][ok] ** 2)
                curves = []
                for sig, w in shifts:
                    k = int(round(abs(sig)))
                    curves.append({
                        "h": hist(x, edges, w[sel][ok]),
                        "color": SIGMA_COLORS.get(k, "tab:gray"),
                        "ls": "-" if sig > 0 else "--",
                        "label": f"{sig:+g}$\\sigma$"})
                if args.show_off:
                    curves.append({"h": hist(x, edges, rest[sel][ok]),
                                   "color": "tab:red", "ls": ":",
                                   "label": "dial off"})
                curves.sort(key=lambda c: c["label"])

                label = short(dial, args.short_labels)
                title = (args.title or "{dial}\n{sample}").format(
                    dial=label, var=var,
                    sample=CONT_LABEL[cont] + SAMPLE_LABEL[sample]
                    + INTERACTION_LABEL[args.interaction])
                out = os.path.join(args.outdir, tag, var,
                                   f"{dial}.{args.format}")
                draw(curves, cv_h, cv_e2, edges, args,
                     VARIABLES[var][0], title, out,
                     logx=var in args.logx_vars,
                     uniform=args.uniform_axis and (var in custom or special))
                n_plots += 1
        wr.cache.pop(dial, None)

    print(f"Saved {n_plots} plot(s) under {args.outdir}/<[fc_|pc_]sample>/<variable>/")


if __name__ == "__main__":
    main()
