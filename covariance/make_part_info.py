#!/usr/bin/env python3
"""
make_part_info.py

Per-event pandora-particle summary for the FIRST pandora interaction of each
cafTree entry, written to a small .npz file that the plotting scripts read
(--part-info):

    n_particles  : number of particles with origRecoObjType != kHitCollection (3)
                   (-1 if the entry has no pandora interaction)
    walldist_min : smallest walldist [cm] over those particles (closest
                   distance to a detector wall); NaN if no particle has one
    n_entries    : number of cafTree entries (for a consistency check)

Why a separate step: common.ixn.pandora.part.pandora is stored "memberwise"
in the CAF, which uproot cannot read. ROOT can, so this script uses PyROOT +
RDataFrame. Run it with a Python that can `import ROOT` (e.g. after setting up
ROOT on the GPVM); it does NOT need the plotting scripts' .venv packages
other than numpy.

Class definitions for caf::StandardRecord are needed. By default they are
generated from the CAF file itself (TFile::MakeProject) and compiled once
into --project-dir (with ACLiC if rootcling/rootcint is not available, e.g.
for the pip ROOT package; only g++ is needed), then reused; duneanaobj does
not have to be installed. Alternatively pass --dict-lib with an existing duneanaobj
dictionary library.

Usage:
    python make_part_info.py                         # default CAF -> part_info.npz
    python make_part_info.py --caf other.root --out other_part_info.npz
    python make_part_info.py --threads 8
    python make_part_info.py --dict-lib libduneanaobj_StandardRecord_dict.so
"""
import argparse
import os
import shutil
import sys
import time

import numpy as np

DEFAULT_CAF = "/exp/dune/data/users/pgranger/caf_new_sum.2.6M_weighted.root"

HELPERS = r"""
#include <cmath>
#include <limits>
// Templated so they accept the RVec RDataFrame hands over. 3 = kHitCollection.
template <class V> float partinfo_walldist_min(const V& ixn) {
  float m = std::numeric_limits<float>::quiet_NaN();
  if (ixn.size() == 0) return m;
  for (auto const& p : ixn[0].part.pandora) {
    if ((int)p.origRecoObjType == 3) continue;
    float w = p.walldist;
    if (std::isnan(w)) continue;
    if (std::isnan(m) || w < m) m = w;
  }
  return m;
}
template <class V> int partinfo_n_particles(const V& ixn) {
  if (ixn.size() == 0) return -1;
  int n = 0;
  for (auto const& p : ixn[0].part.pandora)
    if ((int)p.origRecoObjType != 3) n++;
  return n;
}
"""


def build_classes(ROOT, caf, proj, name):
    """Generate the caf:: classes from the file's streamer info and compile
    them. Returns the shared library path.

    1. TFile::MakeProject writes headers, LinkDef and the class source. Its
       own build script (MAKEP) needs `rootcling`/`rootcint`, which e.g. the
       pip ROOT package does not provide; if it fails we carry on.
    2. ACLiC compiles the class source and makes the dictionary in-process,
       so only a C++ compiler is needed.
    """
    print(f"generating caf classes from {caf} into {proj} "
          f"(one-off; needs a C++ compiler) ...")
    f = ROOT.TFile.Open(caf)
    f.MakeProject(proj, "*", "recreate+")
    f.Close()
    lib = os.path.join(proj, name + ".so")
    if os.path.exists(lib):
        return lib                                  # MAKEP worked
    src = os.path.join(proj, name + "ProjectSource.cxx")
    linkdef = os.path.join(proj, name + "LinkDef.h")
    if not (os.path.exists(src) and os.path.exists(linkdef)):
        sys.exit(f"MakeProject did not write {src} / {linkdef}")
    print("MAKEP could not build the library (no rootcling/rootcint); "
          "compiling with ACLiC instead ...")
    lines = [l for l in open(src).read().splitlines()
             if not (l.startswith("#include")
                     and (l.endswith('Dict.cxx"') or l.endswith('LinkDef.h"')))]
    acl = os.path.join(proj, name + "_aclic.cxx")
    with open(acl, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    shutil.copy(linkdef, os.path.join(proj, name + "_aclic_linkdef.h"))
    if not ROOT.gSystem.CompileMacro(acl, "kO"):
        sys.exit("ACLiC could not compile the generated classes; check that "
                 "a C++ compiler (g++) is on PATH, or use --dict-lib")
    return os.path.join(proj, name + "_aclic_cxx.so")


def load_classes(ROOT, caf, args):
    if args.dict_lib:
        for lib in args.dict_lib:
            if ROOT.gSystem.Load(lib) < 0:
                sys.exit(f"could not load {lib}")
        print(f"loaded dictionary: {', '.join(args.dict_lib)}")
        return
    proj = os.path.abspath(args.project_dir)
    name = os.path.basename(proj)
    libs = [os.path.join(proj, name + ".so"),
            os.path.join(proj, name + "_aclic_cxx.so")]
    lib = next((l for l in libs if os.path.exists(l)), None)
    if lib is None:
        lib = build_classes(ROOT, caf, proj, name)
    if ROOT.gSystem.Load(lib) < 0:
        sys.exit(f"could not load {lib}")
    ROOT.gInterpreter.Declare(
        f'#include "{os.path.join(proj, name + "ProjectHeaders.h")}"')
    print(f"loaded generated classes: {lib}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                allow_abbrev=False)
    p.add_argument("--caf", default=DEFAULT_CAF)
    p.add_argument("--tree", default="cafTree")
    p.add_argument("--out", default="part_info.npz")
    p.add_argument("--project-dir", default="caf_classes",
                   help="where MakeProject writes/reuses the generated classes")
    p.add_argument("--dict-lib", nargs="+", default=None,
                   help="use these dictionary libraries instead of MakeProject")
    p.add_argument("--threads", type=int, default=0,
                   help="RDataFrame implicit multithreading (0 = off)")
    p.add_argument("--max-events", type=int, default=None)
    args = p.parse_args()

    try:
        import ROOT
    except ImportError:
        sys.exit("this script needs PyROOT (`import ROOT`); run it with the "
                 "Python of a ROOT installation")
    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kError      # silence "no dictionary" noise

    load_classes(ROOT, args.caf, args)
    ROOT.gInterpreter.Declare(HELPERS)
    if args.threads:
        ROOT.EnableImplicitMT(args.threads)

    df = ROOT.RDataFrame(args.tree, args.caf)
    n_entries = int(df.Count().GetValue())
    if args.max_events:
        if args.threads:
            sys.exit("--max-events cannot be combined with --threads")
        df = df.Range(args.max_events)
    cols = {str(c) for c in df.GetColumnNames()}
    col = next((c for c in ("rec.common.ixn.pandora", "common.ixn.pandora")
                if c in cols), None)
    if col is None:
        sys.exit("no rec.common.ixn.pandora column in the tree")

    t0 = time.time()
    out = (df.Define("walldist_min", f"partinfo_walldist_min({col})")
             .Define("n_particles", f"partinfo_n_particles({col})")
             .AsNumpy(["walldist_min", "n_particles"]))
    wd = np.asarray(out["walldist_min"], dtype=float)
    npart = np.asarray(out["n_particles"], dtype=float)
    npart[npart < 0] = np.nan
    np.savez(args.out, walldist_min=wd, n_particles=npart,
             n_entries=n_entries, caf=os.path.abspath(args.caf))
    print(f"wrote {args.out}: {len(wd)} entries in {time.time() - t0:.0f} s; "
          f"walldist defined for {np.isfinite(wd).sum()}, "
          f"> 20 cm for {(wd > 20).sum()}, <= 20 cm for {(wd <= 20).sum()}")


if __name__ == "__main__":
    main()
