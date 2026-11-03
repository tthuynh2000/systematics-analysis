# DUNE atmospheric (ANOA) cross-section systematics tools

Python scripts to study cross-section systematic dials for the DUNE
atmospheric-neutrino oscillation analysis:

| Script | What it does |
|---|---|
| `plot_cv_weights.py` | Histograms the per-event weight of a group of dials at their central value (or any σ) |
| `plot_syst_spectra.py` | Reco and true spectra (ν<sub>e</sub> / ν<sub>μ</sub> / NC samples) at CV and ±nσ for each dial, with ratio-to-CV panels |
| `syst_covariance.py` | Per-dial covariance matrices from ±σ universes, their sum, correlation matrices and per-bin error bands |
| `plot_reco_cosz.py` | Reco cos θ<sub>z</sub> from the shower (`heshw`) and track (`lngtrk`) directions for ν<sub>e</sub>-like / ν<sub>μ</sub>-like samples, weighted by the base weight × dial CVs (§5.2) |
| `plot_syst_spectra_josh.py` | Same spectra as `plot_syst_spectra.py`, with **Josh's** selection, reconstruction and one-dial weights (§6d) |
| `syst_covariance_josh.py` | Same covariance matrices / bands as `syst_covariance.py`, with **Josh's** conventions (§6d) |
| `make_part_info.py` | **Run once with ROOT.** Per-event closest particle wall distance (for FC/PC) and exact particle count from the memberwise `part.pandora` branch → `part_info.npz` |
| `check_weights.py` | Optional diagnostic: statistics of the CAF `weights` branches and how `final_oscillated_w` relates to flux / xsec / oscillation branches |

All scripts must sit in the **same folder**: `plot_syst_spectra.py` takes
its dial groups from `plot_cv_weights.py`, and `syst_covariance.py` takes its
loading, weighting and binning from `plot_syst_spectra.py`. The selection, CV
definition and binning are therefore identical across all plots.

---

## 1. Setup

```bash
git clone <this repo URL> anoa-xsec-systs
cd anoa-xsec-systs
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Requires Python ≥ 3.8. After activating the venv, `python` and `python3` both
point to the venv interpreter.

## 2. Input files

Default paths, used when `--caf` / `--weights` are not given:

| | Default path | Contents used |
|---|---|---|
| CAF | `/exp/dune/data/users/pgranger/caf_new_sum.2.6M_weighted.root` | `cafTree` (`rec/mc/mc.nu`, `rec/common/common.ixn.pandora`, `rec/fd/fd.hd.pandora.npfps`) and `weights` (newest cycle) |
| Weights | `/exp/dune/data/users/tthuynh/merged_new_systs_wsbn_hotfixed_full_header.root` | `SystWeights` (one weight vector per event per dial), `systsHeader` (`name`, `cv`, `variations`, `id`, `isCorrection`) |

The defaults are set as `DEFAULT_CAF` / `DEFAULT_WEIGHTS` near the top of each
script. The CAF and weight files must match **entry by entry**: same number of
events, same order.

**Weight layout.** For a dial with `variations = [0, 1, -1, 2, -2, 3, -3]`,
the `SystWeights` branch of that dial holds, per event, a vector of 7 weights
in that same order. The script finds the weight at a given σ by looking up the
index of σ in `variations`. The central value is the entry at index
`variations.index(cv)`, with `cv` taken from `systsHeader/cv`. Correction dials
(0 or 1 entries in `variations`) use element 0.

## 3. Quick start

```bash
# once, with a Python that has PyROOT (not needed for plain FC/PC-free runs):
python make_part_info.py                 # -> part_info.npz (FC/PC + exact nParticles)
```

```bash
# Inspect the dials: cv, variations, which index is the CV
python plot_cv_weights.py --list

# 1) CV weight distributions for a group of dials
python plot_cv_weights.py --group CCQE_Axial_MvA_PCA --nbins 50 --logy --legend-stats --short-labels

# 2) Spectra at CV and +/-1,2,3 sigma, DUNE atm binning, log x
python plot_syst_spectra.py --group FSI_Pion --binning atm --logx --short-labels --outdir spectra_fsi

# 3) Covariance matrices + error band from +/-1 sigma universes
python syst_covariance.py --group FSI_Pion --binning atm --logx --per-dial-plots --short-labels --outdir cov_fsi
```

The CAF has about 2.6 M events. Add `--max-events 200000` for quick checks.

---

## 4. `plot_cv_weights.py`: weight distributions

For each selected dial it takes the per-event weight at the dial's `cv` (or at
`--sigma X`) and histograms it. The x axis is the weight, binned evenly from
min to max (`--nbins`); the y axis is the number of events.

```bash
python plot_cv_weights.py [files ...] --group GENIE_RES --skip-trivial
python plot_cv_weights.py --dials "*FrAbs*" "*FrCEx*" --layout grid --logy
python plot_cv_weights.py --group MEC_XSecShape --sigma 1          # +1 sigma weights instead of CV
```

| Option | Meaning |
|---|---|
| `--list` | Print every dial with its `cv`, `isCorrection`, index of the target value, and variations |
| `--group / --dials / --regex / --exclude` | Dial selection (groups in §8, globs like `"ZExpPCA*"`) |
| `--sigma X` | Plot weights at variation `X` instead of the header `cv` |
| `--missing drop\|one` | Events with no entry at that index: drop (default) or set weight to 1 |
| `--skip-trivial` | Hide dials whose weights are all exactly 1 (most GENIE multisigma dials at CV) |
| `--nbins`, `--range` | Binning |
| `--layout overlay\|grid` | One axis for all dials, or one panel per dial (`--ncols`, `--sharex`) |
| `--logx`, `--logy` | Log axes (`--logx` uses log-spaced bins and drops weights ≤ 0) |
| `--density`, `--histtype`, `--cmap`, `--linewidth`, `--alpha` | Style |
| `--legend-stats`, `--short-labels`, `--no-legend`, `--legend-*` | Legend |
| `--title`, `--xlabel`, `--ylabel`, `--vline 1`, `--grid-lines`, `--figsize` | Labels and decorations |
| `-o file.png`, `--dpi`, `--show` | Output |

If several dials have identical weights event by event, their histograms
overlap exactly. The script prints a note listing them; `--layout grid` shows
them separately.

---

## 5. `plot_syst_spectra.py`: spectra with dial variations

For every selected dial × sample × variable it draws one figure:

* **Top panel:** CV (black), +nσ (solid) and −nσ (dashed), with one colour per n. Optional extras:
  * the "dial off" curve (`--show-off`);
  * the CV MC-statistical band (`--stat-band`).
* **Bottom panel:** ratio to CV.

Output: `<outdir>/<sample>/<variable>/<dial>.png`.

```bash
python plot_syst_spectra.py --dials "*MaCCRES" --variables reco_Elep true_Elep \
    --samples numu --binning atm --uniform-axis --show-off --stat-band
python plot_syst_spectra.py --group CCQE_Axial_Deut_PCA --sigmas 1 --ratio-logy
```

### 5.1 Definitions

The variable definitions follow [pgranger23/systematics-analysis `systs.py`](https://github.com/pgranger23/systematics-analysis/blob/main/systs.py).
The sample selection uses CVN thresholds (below). All reco quantities come
from `common.ixn.pandora`.

**Sample selection (default `--selection cuts`).** With
`pass_μ = cvn.numu > 0.56` and `pass_e = cvn.nue > 0.55`:

| pass_μ | pass_e | Sample |
|---|---|---|
| no | no | NC |
| yes | no | ν<sub>μ</sub> |
| no | yes | ν<sub>e</sub> |
| yes | yes | ν<sub>μ</sub> if `cvn.numu > cvn.nue`, otherwise ν<sub>e</sub> |

* A score exactly at the cut does not pass, and a missing (NaN) score never passes.
* `--cvn-numu-cut` / `--cvn-nue-cut` change the thresholds; the defaults are `CVN_NUMU_CUT` / `CVN_NUE_CUT` in `plot_syst_spectra.py`.
* `--selection argmax` restores the `systs.py` CVN argmax of (nue, numu, nc).
* Both scripts print the number of events in each reco sample.

| Quantity | Definition |
|---|---|
| Reco flavour / sample | CVN threshold selection above → ν<sub>e</sub>, ν<sub>μ</sub> or NC sample |
| `reco_E` | ν<sub>μ</sub>: `Enu.lep_calo`; otherwise `Enu.e_calo` |
| `reco_Ehad` | ν<sub>μ</sub>: `Enu.mu_had`; otherwise `Enu.e_had` |
| `reco_Elep` | `reco_E − reco_Ehad` |
| `reco_cosz` (default `--cosz mach3`) | −d<sub>y</sub>/\|d\| (negative means upgoing), with the direction d chosen per event as in MaCh3 (rows below) |
| `reco_cosz`: nParticles | Pandora particles of the first interaction excluding the unassociated-hits object (`part.pandora.origRecoObjType ≠ kHitCollection = 3`), i.e. NbPFPs − 1. `--nparticles-from`: `type` (that count), `npandora` (`part.npandora` − 1), `npfps` (`fd.hd.pandora.npfps` − 1), `auto` (default: `type` if readable, else `npandora`) |
| `reco_cosz`: d = `dir.calo` | if nParticles < 2, or `Enu.calo` ≥ 1.3 GeV or NaN ("DirectionRecHits") |
| `reco_cosz`: otherwise | `dir.heshw` for the ν<sub>e</sub>-like sample ("DirectionRecNuEPfps"), `dir.lngtrk` for ν<sub>μ</sub>-like and NC ("DirectionRecNuMuPfps") |
| `reco_cosz`: which events are ν<sub>e</sub>-like | Follows `--selection`. MaCh3's `isELike` uses its own rule (ν<sub>μ</sub> wins when both CVN cuts pass), so add `--selection mach3` to reproduce MaCh3 exactly; the default `cuts` gives such events to the larger score |
| `reco_cosz` with `--cosz calo` | −`dir.calo.y` for all samples (previous default) |
| `reco_cosz` with `--cosz systs` | Old `systs.py` estimator: ν<sub>μ</sub> → `dir.lngtrk.y`; ν<sub>e</sub> and NC → `dir.heshw.y`; falls back to `dir.heshw.y` when `npfps < 3`, `Enu.e_calo > 1.3 GeV`, or the flavour estimator is 0 |
| `true_E` | `mc.nu.E` |
| `true_Elep` | E·(1 − y), y = `mc.nu.inelasticity` |
| `true_Ehad` | E·y |
| `true_cosz` | −p<sub>y</sub>/E of the neutrino |
| `reco_y` | `reco_Ehad / reco_E` (NaN if `reco_E` ≤ 0) |
| `true_y` | `mc.nu.inelasticity` |

True-variable plots are split by the same CVN reco samples.

**Combined samples.** `--samples` takes the samples `nue`, `numu` and `nc`,
plus two combinations:
* `nue+numu`: the ν<sub>e</sub> and ν<sub>μ</sub> samples added into one spectrum;
* `all`: ν<sub>e</sub> + ν<sub>μ</sub> + NC.

Each event keeps the reco energies of its own sample (e.g. `Enu.lep_calo` if
it was selected as ν<sub>μ</sub>). The combined spectrum and universes are
therefore exactly the sum of the separate samples. Combined samples work in
both `plot_syst_spectra.py` and `syst_covariance.py` (including `--joint`).

```bash
python plot_syst_spectra.py --dials "*MaCCRES" --sigmas --samples nue+numu --binning atm --uniform-axis   # CV only
python syst_covariance.py --group GENIE_RES --samples nue+numu --binning atm --logx
```

**Event selection (default):**
* true neutrinos only (PDG > 0; `--include-nubar` keeps antineutrinos);
* quality cut (`--no-quality-cut` disables it):
  * a pandora interaction exists;
  * `npfps ≥ 2`;
  * CVN is defined;
  * the reco direction used for cos θ<sub>z</sub> is non-zero and finite (with `--cosz mach3`, the chosen vector has \|d\| > 0; `dir.calo.y` with `calo`; `dir.heshw.y` with `systs`);
* **True interaction type** (`mc.nu.iscc`), with `--interaction` (also in `syst_covariance.py`):
  * `all` (default): CC + NC;
  * `cc`: true CC only;
  * `nc`: true NC only.

  `--cc-only` is kept as a shortcut for `--interaction cc`. The choice is added to the plot titles (e.g. "ν<sub>e</sub> + ν<sub>μ</sub> samples, true CC"). This cut is on the *true* interaction; the NC *sample* (`--samples nc`) is the reco CVN selection. The two combine, e.g. `--samples nue+numu --interaction nc` shows true-NC background in the ν<sub>e</sub>+ν<sub>μ</sub> samples.

Each entry is assumed to hold one neutrino and one reco interaction, as in
`systs.py`; the first one is used and a warning is printed if there are more.

### 5.2 Weights

All weight inputs come from the CAF `weights` tree. uproot reads its newest
cycle (`weights;7`), and the systematics file supplies only the dial weights.

```
base (default --base-weight flux_xsec) = xsec × flux of the event's own TRUE flavour, unoscillated
                                           |pdg| = 12 → flux_nue,  |pdg| = 14 → flux_numu,  other → 0
base (--base-weight final)             = final_oscillated_w
CV         = base × Π_g w_g(cv)               one factor per CV group g (see below)
dial X @ v = base × Π_{g≠g(X)} w_g(cv) × w_X(v)  (all other dials stay at CV)
X off      = base × Π_{g≠g(X)} w_g(cv)            (--show-off)
```

**Shared b-scan zero response.** The dials `<provider>_b1 … _bN` (e.g.
`ZExpPCAWeighter_SBN_v3_MvA_b1..b4`, `…_Deut_b1..b4`) are scans of one
provider. They share the same zero-displacement model, so their CV response is
a single correction:
* it is applied **once** per provider, not once per scan;
* varying `MvA_b2` replaces that one factor by `w_b2(v)`;
* the script warns if the scans' CV responses are not identical;
* `--independent-b-scans` restores multiplying every scan.

**Raw stored responses** (default):
* negative dial weights are kept, so signed rates are retained;
* a NaN/inf response is kept as NaN, which makes **only the bin that event falls in** missing (drawn as a gap; in `syst_covariance.py` its matrix row and column are NaN);
* an event with no entry at the requested index gets 1;
* a NaN base weight is set to 0;
* histograms are filled with `np.bincount`, because `np.histogram` would spread a NaN into every following bin;
* `--clip-responses` restores the old handling (negative → 0, nonfinite → 1).

`--no-cv` drops the other dials' CV weights, which reproduces `systs.py`.

### 5.3 Main options

| Option | Meaning |
|---|---|
| `--selection cuts\|mach3\|argmax`, `--cvn-numu-cut 0.56`, `--cvn-nue-cut 0.55` | Sample selection (§5.1; `mach3` = ν<sub>μ</sub> wins when both cuts pass); also in `syst_covariance.py` |
| `--cosz mach3\|calo\|systs\|detsys` | Reco cos θ<sub>z</sub> estimator (§5.1; default `mach3`; `detsys` = ν<sub>μ</sub> → `lngtrk`, ν<sub>e</sub>/NC → `heshw`, no fallback); also in `syst_covariance.py` |
| `--detsys-compatible` | Match Claire's detsys plots (§5.4); also in `syst_covariance.py` |
| `--interaction all\|cc\|nc` | True interaction type: CC + NC (default), CC only, NC only; also in `syst_covariance.py` |
| `--base-weight flux_xsec\|final`, `--clip-responses`, `--independent-b-scans` | Base weight and dial-response handling (§5.2); also in `syst_covariance.py` and `plot_reco_cosz.py` |
| `--sigmas 1 2 3`, `--only-up`, `--only-down` | Which shifts to draw |
| `--show-off` | Draw the varied dial switched off |
| `--samples nue numu nc nue+numu all`, `--variables ...` | What to plot (default samples: nue numu nc; default variables: reco and true Elep, Ehad, cosz) |
| `--sigmas` (no values) | CV only, no ±σ curves |
| `--stat-band` | CV MC-statistical uncertainty, √Σw² per bin |
| `--ratio-range lo hi`, `--ratio-logy` | Ratio panel range / log scale (auto range fits all curves) |
| `--logy`, `--logx`, `--logx-vars v1 v2` | Log axes (cos θ<sub>z</sub> always stays linear) |
| `--title "{dial} {sample}"`, `--short-labels`, legend/font options | Labels |

---

### 5.4 Comparing with Claire's detsys plots (`--detsys-compatible`)

Claire's detector-systematics tools are in
[DUNE/atmospherics-tools, branch `cdalmazz/feature/cAFComparisonTools`, `detsys/`](https://github.com/DUNE/atmospherics-tools/tree/origin/cdalmazz/feature/cAFComparisonTools/detsys).
`--detsys-compatible` switches every option that differs from her
`Reader.cxx` / `SampleManager.cxx`:

| | detsys (Claire) | `--detsys-compatible` sets |
|---|---|---|
| Reco cos θ<sub>z</sub> | ν<sub>μ</sub>: −`lngtrk.y`; ν<sub>e</sub>, NC: −`heshw.y`; no fallback | `--cosz detsys` |
| CVN tie-break | ν<sub>μ</sub> if `cvn.numu > 0.56`, else ν<sub>e</sub> if `cvn.nue > 0.55`, else NC | `--selection mach3` |
| Quality cut | none: `dir = 0` events kept at cos θ<sub>z</sub> = 0, i.e. in the [0, 0.2) bin | `--no-quality-cut` |
| Pandora interactions | exactly one, otherwise reco = −999 (underflow) | reco values only for events with exactly one interaction |
| ν̄ | included | `--include-nubar` |
| Y axis | divided by bin width, "Number of Events/Bin Width" | `--per-width` + that label |

**Not changed:**
* **Weights** stay the base weight × dial CVs (§5.2). Claire computes hers from her own flux tables, oscillation code and exposure, with no xsec CV, so the normalisation will differ.
* **Input MC** stays your CAF; hers is the `atm-ref` production.

In `syst_covariance.py` the matrices stay in events; only the band plot is
divided by bin width.

Her RecoCosZ plot fills ν<sub>e</sub>, ν<sub>μ</sub> and NC into one histogram,
which corresponds to `--samples all`. `nue+numu` leaves NC out.

```bash
# combined nue+numu (and, for her exact plot, all), CV only, detsys conditions
python plot_syst_spectra.py --dials "*MaCCRES" --sigmas --detsys-compatible \
    --samples nue+numu all --variables reco_cosz --binning atm --outdir detsys_cv

# same, with +/-1 sigma of a dial group, and the covariance error band
python plot_syst_spectra.py --group FSI_Pion --sigmas 1 --detsys-compatible \
    --samples nue+numu --variables reco_cosz --binning atm
python syst_covariance.py --group FSI_Pion --detsys-compatible \
    --samples nue+numu --variables reco_cosz --binning atm
```

---

## 6. `syst_covariance.py`: covariance matrices and error bands

### 6.1 Method

N is the vector of binned event counts (bins of the chosen observables). For
**one dial at a time**, the other dials stay at CV, and the universes are the
dial at cv − s and cv + s (default s = 1):

$$E^{(d)}_{ij} = \frac{1}{N_{\rm univ}}\sum_{\alpha}\left(N_i^{\alpha}-N_i^{\rm CV}\right)\left(N_j^{\alpha}-N_j^{\rm CV}\right)$$

The dials are treated as independent and **added**:

$$E_{ij} = \sum_{d} E^{(d)}_{ij} \;\;(+\ \text{optional statistical diagonal})$$

The **error band** in each bin is σ<sub>i</sub> = √E<sub>ii</sub>, drawn as
CV ± σ<sub>i</sub>. The ratio panel shows 1 ± σ<sub>i</sub>/N<sub>i</sub>.

Notes:
* Correction dials (one variation) have no universes. They are skipped here but still enter the CV.
* With ±1σ universes, a dial's band edges coincide with its ±1σ spectra. This is a quick check that the covariance is built correctly.
* `--sigmas 1 2 3` also averages the ±2σ and ±3σ universes in with 1/N<sub>univ</sub>. The result is then no longer a 1σ covariance, and a warning is printed.
* Matrices are built from **event counts**, not a cross section. The slide formula with flux Φ, N<sub>targets</sub> and response R would need those inputs.

### 6.2 Usage

```bash
# one matrix per (sample, variable)
python syst_covariance.py --group GENIE_RES --binning atm --logx --per-dial-plots

# one joint matrix (cross-sample and cross-variable correlations), + MC stat
python syst_covariance.py --group FSI_Pion --binning atm --samples numu nue \
    --variables reco_Elep reco_cosz --joint --add-stat mc
```

| Option | Meaning |
|---|---|
| `--sigmas 1` | Universes per dial: cv ± s for each s |
| `--add-stat none\|mc\|poisson` | Add Σw² (MC stat) or N<sub>CV</sub> (Poisson) on the diagonal |
| `--joint` | One matrix over all (sample, variable) blocks instead of one per block |
| `--per-dial-plots` | Also plot each dial's own covariance, correlation and band in `<outdir>/dials/` |
| `--per-width`, `--ylabel` | Band plot divided by bin width (matrices unchanged) / its y label |
| `--detsys-compatible`, `--cosz`, `--selection` | Same as in the spectra script (§5.1, §5.4) |
| `--band-universes 5` | On the total band plot, overlay the ±σ spectra of the N largest dials (0 = none) |
| `--logx`, `--logx-zero x` | Physical log x (and log matrix axes) for energy variables, so bins keep their true relative widths. A lower edge of 0 is drawn one log-width below the next edge and still labelled "0" |
| `--top-dials 8` | Dials shown individually in the uncertainty breakdown |
| `--annotate`, `--cmap`, font/legend options | Style |
| `--no-root` | Write only `.npz` |

### 6.3 Outputs (per matrix, `<outdir>/<sample>_<variable>_*` or `joint_*`)

| File | Content |
|---|---|
| `*_cov.png` | Total covariance E<sub>ij</sub> |
| `*_fraccov.png` | Fractional covariance E<sub>ij</sub>/(N<sub>i</sub>N<sub>j</sub>) |
| `*_corr.png` | Correlation E<sub>ij</sub>/√(E<sub>ii</sub>E<sub>jj</sub>) |
| `*_fracerr.png` | √E<sub>ii</sub>/N<sub>i</sub> per bin: total and the largest dials |
| `*_band.png` | CV ± √E<sub>ii</sub> with ±σ universe curves and a ratio panel |
| `*.npz` | `cv`, `cov`, `cov_syst`, `cov_stat`, `frac_cov`, `corr`, `edges__<sample>__<var>`, `cov_dial__<dial>`, `universes__<dial>`, `universe_sigmas__<dial>`, `blocks`, `block_offsets` |
| `*.root` | `cv`, `cov`, `cov_syst`, `frac_cov`, `corr` (TH1/TH2 on a bin-index axis), `dials/cov_<dial>`, `blocks/cv_<sample>_<var>` (physical edges) |

The script prints the matrix size, the total CV, the minimum eigenvalue, the
rank (each dial adds at most 2 with ±1σ) and the symmetry check for every
matrix.

---

## 6a. `plot_reco_cosz.py`: shower vs track reco cos θ<sub>z</sub>

Standalone (it does not need the other scripts). For each sample (ν<sub>e</sub>-like
blue, ν<sub>μ</sub>-like orange) it plots reco cos θ<sub>z</sub> of the first
pandora interaction.

* **`--cosz mach3` (default):** the MaCh3 algorithm, identical to §5.1. The direction is `dir.calo` if there are fewer than 2 particles (hit collection excluded) or `Enu.calo` ≥ 1.3 GeV or NaN. Otherwise it is `dir.heshw` for ν<sub>e</sub>-like and `dir.lngtrk` for ν<sub>μ</sub>-like, giving cos θ<sub>z</sub> = −d<sub>y</sub>/\|d\|. The legend gives the fraction of events that used `dir.calo`.
* **`--cosz pfp`:** the raw −`dir.heshw.y` (ν<sub>e</sub>-like) / −`dir.lngtrk.y` (ν<sub>μ</sub>-like), with no `dir.calo` switch and no normalisation. `--show-both` also overlays the other estimator (dashed).

With `--cvn`, "ν<sub>e</sub>-like" follows `--selection` (default `mach3`, the MaCh3 rule), as MaCh3's `isELike` does. Values outside [−1, 1] (CAF sentinels) and NaN are dropped.

**Weights (y axis):**

```
w = base × Π_g w_g(cv)     (same base weight, raw responses and b-scan rule as §5.2)
```

* `--cv-dials` restricts which dials enter the product (globs, default all).
* `--base-weight flux_xsec|final`, `--clip-responses` and `--independent-b-scans` work as in §5.2.
* `--no-dial-cv` uses the base weight only; `--unweighted` gives plain counts.
* A bin touched by a nonfinite response is drawn as a gap, and the legend notes the missing bins.
* The legend shows N and Σw per curve, and the terminal prints Σw per sample, including NC-like.

**Samples:**
* `--cvn FILE... --syst FILE...`: classify with CVN; one systematics file per CAF file, same order.
  * `--selection mach3` (default): ν<sub>μ</sub> if `cvn.numu > 0.56`, else ν<sub>e</sub> if `cvn.nue > 0.55`, else NC.
  * `--selection analysis`: the threshold rule of §5.1, where the larger score wins when both pass.
* `--nue FILE... --nue-syst FILE... --numu FILE... --numu-syst FILE...`: pre-selected files.
* No sample option: `--cvn DEFAULT_CAF --syst DEFAULT_WEIGHTS`.

```bash
python plot_reco_cosz.py                                     # default files, weighted
python plot_reco_cosz.py --cvn caf.root --syst syst.root --bins 40 --out cosz.pdf
python plot_reco_cosz.py --cvn caf.root --unweighted         # plain counts
```

---

## 6b. `check_weights.py`: weight diagnostic (optional)

The plotting scripts use xsec × flux (or `final_oscillated_w` with `--base-weight final`). This script only shows
how it relates to the other branches in the `weights` tree.

```bash
python check_weights.py                      # default CAF
python check_weights.py --caf other.root --max-events 500000
```

| Section | What it reports |
|---|---|
| [1] | The `weights` cycles (e.g. `;6`, `;7`), which one uproot uses, and whether they differ |
| [2] | Per-branch N, NaN/inf, negatives, zeros, min/mean/max |
| [3] | How well each candidate formula (including `F = flux_nue·osc_e + flux_numu·osc_mu` and `xsec·F`) reproduces `final_oscillated_w`: median and max relative difference, fraction agreeing to 10⁻⁵, ratio median and 5–95 % spread. Prints the closest |
| [4] | `weights.xsec` vs `mc.nu.xsec` / `genweight` / `xsec_cvwgt`; `nue_w`/`numu_w` vs `flux_*` and `xsec·flux_*` |
| [5] | Total weighted events for each formula |

---

## 6c. FC / PC containment (`--containment`)

**Definition.** `walldist` is a per-particle quantity ("closest distance to a
detector wall [cm]"). For each event, `make_part_info.py` takes the smallest
`walldist` among the particles of the first pandora interaction, excluding the
unassociated-hits object (and particles with NaN `walldist`):

| | Rule (`--walldist-cut 20`) |
|---|---|
| FC | `walldist_min > 20 cm` |
| PC | `walldist_min ≤ 20 cm` (exactly 20 → PC) |
| neither | no particle with a defined `walldist`: excluded from both FC and PC; kept in `all` |

**Running.** `--containment fc pc` (or `all`, the default) in
`plot_syst_spectra.py` and `syst_covariance.py`.
* Each value gives its own plots or matrices: `fc_numu/`, `pc_nue_reco_E_*`, ….
* Titles start with "FC" or "PC".
* With `--joint`, all containments, samples and variables go into one matrix, which includes the FC–PC correlations.
* The scripts print the FC/PC/undefined event counts.

**Why a separate ROOT step.** `common.ixn.pandora.part.pandora` is stored
memberwise, which uproot cannot read. `make_part_info.py` reads it with PyROOT
+ RDataFrame:
* it generates the `caf::` classes from the CAF's own streamer info (`TFile::MakeProject`), so duneanaobj is not needed; `--dict-lib` uses an existing dictionary instead;
* it compiles them once into `caf_classes/` and reuses them afterwards. If ROOT's own build script fails because `rootcling`/`rootcint` is missing (as with the pip `ROOT` package), it compiles with ACLiC instead, which needs only `g++`. The `rootcint: command not found` / `ProjectDict.cxx` errors printed before that fallback are expected;
* it writes `part_info.npz` with `walldist_min`, `n_particles` and `n_entries`;
* the plotting scripts read `--part-info part_info.npz` (the default; used if the file exists) and check that it was made from a CAF with the same number of entries;
* the same file also gives the **exact** type-based nParticles for the MaCh3 cos θ<sub>z</sub>, replacing the `part.npandora − 1` fallback.

```bash
python make_part_info.py --threads 8                     # needs PyROOT + a C++ compiler (first run only)
python plot_syst_spectra.py --group CCQE_Axial_Deut_PCA --containment fc pc \
    --samples nue numu --variables reco_E reco_y reco_cosz --uniform-axis
python syst_covariance.py --group CCQE_Axial_Deut_PCA --containment fc pc \
    --samples nue numu --variables reco_E reco_y reco_cosz --logx
```

---

## 6d. Josh's conventions: `plot_syst_spectra_josh.py`, `syst_covariance_josh.py`

These scripts give the same plots and outputs as the two main scripts. They
reuse the plotting, binning, dial groups and dial reading from
`plot_syst_spectra.py` / `syst_covariance.py`, so keep all of them in one
folder. Only the selection, reconstruction and weights are Josh's. Default
outputs go to `syst_spectra_josh/` and `syst_cov_josh/`, and every title says
"[Josh]". `part_info.npz` from `make_part_info.py` is **required**.

```bash
python plot_syst_spectra_josh.py --group CCQE_Axial_Deut_PCA --containment fc pc \
    --samples cc_nue cc_numu --variables reco_E reco_y reco_cosz --uniform-axis --show-nominal
python syst_covariance_josh.py --group CCQE_Axial_Deut_PCA --containment fc pc \
    --samples cc_nue cc_numu --variables reco_E reco_y reco_cosz --per-dial-plots
```

| | Josh (these scripts) | Yours (main scripts) |
|---|---|---|
| Samples (`--samples`) | **true** channels: `cc_nue` (CC, \|pdg\| = 12), `cc_numu` (CC, \|pdg\| = 14), `nc`, `cc_nue+cc_numu`, `all`; ν and ν̄ together | CVN reco selection; ν only |
| Common selection (cut flow printed) | exactly 1 MC ν; exactly 1 pandora interaction; supported channel; nominal > 0 and finite; true E in [0.1, 100] GeV (`--emin/--emax`); vertex FV (`--fv` box, true or reco vertex; **off unless given**); valid containment; valid true and reco directions. `--skip-cuts` drops any of them | first ν / first interaction; `npfps ≥ 2`; finite CVN; non-zero direction |
| FC / PC | FC: `walldist_min` > 10 cm; PC: 0 ≤ `walldist_min` ≤ 10 cm; missing, negative or > 10000 cm = unknown and removed | > / ≤ `--walldist-cut` (20 cm); only NaN is unknown |
| Reco direction | `dir.calo` if NC, or nParticles < 2, or !(Enu.calo < 1.3); else `heshw` (true CC ν<sub>e</sub>) / `lngtrk` (true CC ν<sub>μ</sub>); −d<sub>y</sub>/\|d\| | same rule, but chosen by the reco sample; NC not forced to `dir.calo` |
| Reco E / E<sub>had</sub> | `e_calo` / `lep_calo` with `e_had` for **both**; NC: E = E<sub>had</sub> = `Enu.calo`; `--reco-energy matched_range` uses `mu_range` / `mu_had` for muons | `mu_had` for the ν<sub>μ</sub> sample |
| Reco y | E<sub>had</sub>/E, 0 if E ≤ 0; NaN for NC | NaN if E ≤ 0 |
| True E<sub>had</sub> / E<sub>lep</sub> / y / cos θ<sub>z</sub> | q0 / E − q0 / inelasticity / −p<sub>y</sub>/\|p\| | E·y / E(1−y) / inelasticity / −p<sub>y</sub>/E |
| Nominal weight | xsec × flux of the true flavour, unoscillated | same |
| Dials | **one-dial**: only the varied dial; no other CVs | all other dials at CV (b-scans once) |
| Responses | nonfinite → 1; clipped to [−9999, 9999]; final weight ≤ 0 → 0 | raw: nonfinite → missing bin; negatives kept |

**Curves and matrices**
* **Spectra:** black is the dial at its cv (Josh's "central/reference" curve). `--show-nominal` adds the no-dial nominal as a dotted line. The y axis is **events / bin width** by default (label "Events / bin width"); `--no-per-width` gives events per bin. The ratio panel is unchanged.
* **Covariance:** each dial's universes are compared with that dial's own cv reference, E = Σ<sub>d</sub> E<sup>(d)</sup>, and the band is drawn around the nominal. Per-dial band plots are drawn around the dial's reference. Each output also gets a `*_refs.npz` with these references.
* **Flow accounting:** the scripts print under/overflow and finite counts per sample and variable. Reco y outside [0, 1] stays in the data but falls outside the histogram edges.
* **Binning:** the FC/PC table (§7) applies to `cc_nue` / `cc_numu`; NC uses the general edges.

**Not reproduced or unknown**
* **Fiducial volume:** Josh's "configured vertex fiducial volume" box isn't known. Give it with `--fv`.
* **Signed angular 68 % width:** this observable is not implemented.
* **Contradiction in his notes:** they say both "nonfinite responses invalidate touched bins" and "nonfinite knots become 1". These scripts follow the verified numerical policy (→ 1).

---
## 7. Binning (spectra and covariance)

Priority, later wins: default uniform → `--nbins` → `--binning` preset →
`--bins var=n,lo,hi` → `--edges var=e0,e1,...`.

**`--binning atm_fcpc` (default).** It starts from the `atm` edges below. For
the FC and PC ν<sub>e</sub> / ν<sub>μ</sub> samples it then uses these
sample-specific edges for E<sub>ν</sub>, y and cos θ<sub>z</sub> (reco and
true):

| Sample | E<sub>ν</sub> [GeV] | y | cos θ<sub>z</sub> |
|---|---|---|---|
| FC ν<sub>e</sub> | 0, 0.2, 0.4, 0.6, 0.8, 1, 1.1, 1.2, 1.35, 1.5, 1.7, 1.95, 2.25, 3.25, 100 | 0, 0.1, 0.2, 0.3, 0.4, 0.5, 1 | −1 … 1 in steps of 0.2 |
| FC ν<sub>μ</sub> | 0, 0.4, 0.6, 0.8, 1, 1.1, 1.2, 1.45, 1.65, 1.9, 100 | 0, 0.1, 0.2, 0.3, 0.4, 0.5, 1 | −1 … 1 in steps of 0.2 |
| PC ν<sub>e</sub> | 0, 0.2, 0.4, 0.6, 0.8, 1.05, 1.4, 2, 3.25, 100 | 0, 0.1, 0.2, 0.3, 0.4, 0.5, 1 | −1 … 1 in steps of 0.2 |
| PC ν<sub>μ</sub> | 0, 0.4, 0.6, 0.8, 1, 1.15, 1.3, 1.45, 1.6, 1.8, 2, 2.2, 2.45, 2.7, 3, 3.35, 3.75, 4.25, 4.85, 6.85, 100 | 0, 0.1, 0.2, 0.3, 1 | −1 … 1 in steps of 0.2 |

* **Other cases** use the general edges: containment `all`, combined samples, NC, and E<sub>lep</sub>/E<sub>had</sub> (which use `atm`). E<sub>ν</sub> and y otherwise get uniform bins.
* **Explicit binning wins:** a variable given with `--edges`/`--bins` overrides this table.
* **Where they live:** the edges are `FCPC_EDGES` in `plot_syst_spectra.py`, and the scripts print the edges used for each containment / sample / variable.
* **`--binning none`** gives uniform bins.

**`--binning atm`** (bin **edges**, applied to both reco and true):

| Variable | Edges | Bins |
|---|---|---|
| E<sub>lep</sub> [GeV] | 0, 0.28, 0.37, 0.50, 0.68, 0.92, 1.3, 1.9, 3.2, 100 | 9 |
| E<sub>had</sub> [GeV] | 0, 0.03, 0.05, 0.09, 0.13, 0.20, 0.31, 0.53, 1.00, 2.5, 100 | 10 |
| cos θ<sub>z</sub> | −1, −0.8, −0.6, −0.4, −0.2, 0, 0.2, 0.4, 0.6, 0.8, 1 | 10 |

**Custom edges:**

```bash
--edges "Elep=[0, 0.28, 0.37, 0.50, 0.68, 0.92, 1.3, 1.9, 3.2, 100]" "true_Ehad=0,0.1,1,5"
```

* `E`, `Elep`, `Ehad` and `cosz` set both `reco_` and `true_`; use the full name for one only.
* Brackets and the unicode "−" are accepted.
* Edges must be strictly increasing.

**Edge convention (numpy):**
* bins are [low, high), except the last bin, which includes its upper edge;
* values outside the edges are not counted.

**Display options for variable-width bins** (spectra script):

| Option | Effect |
|---|---|
| `--uniform-axis` | Every bin the same width on the page, labelled with the real edges |
| `--per-width` | Events per GeV |
| `--logx` | Real widths on a log axis |

---

## 8. Dial groups (`--group`)

Defined in `GROUPS` at the top of `plot_cv_weights.py` (shared by all scripts;
edit freely). Patterns are shell globs (`*`, `?`, `[1-4]`).

| Group | Physics | Patterns |
|---|---|---|
| `CCQE_Axial_GENIE_ZExp` | CCQE axial form factor, GENIE z-expansion | `GENIEReWeight_ICARUS_v2_multisigma_ZExpA[1-4]CCQE` |
| `CCQE_Axial_Deut_PCA` | CCQE axial FF, deuterium-fit PCA (b-scans: CV applied once) | `ZExpPCAWeighter_SBN_v3_Deut_b[1-4]` |
| `CCQE_Axial_MvA_PCA` | CCQE axial FF, MvA PCA (b-scans: CV applied once) | `ZExpPCAWeighter_SBN_v3_MvA_b[1-4]` |
| `CCQE_Vector` | CCQE vector form factors | `GENIEReWeight_ICARUS_v2_multisigma_VecFFCCQEshape` |
| `CCQETemplate_HFToCRPA` | CCQE nuclear-model template | `CCQETemplateReweight_SBN_v3_HFToCRPA_q0bin[0-4]` |
| `CCQETemplate_LFGToHF` | CCQE nuclear-model template | `CCQETemplateReweight_SBN_v3_LFGToHF_q0bin[0-4]` |
| `CCQETemplate_LFGToSF` | CCQE nuclear-model template | `CCQETemplateReweight_SBN_v3_LFGToSF_q0bin[0-4]` |
| `QEInterference` | QE interference | `QEInterference_SBN_v3_QEIntf_dial_[0-5]` |
| `CCQEXSecCorr` | CCQE central-model correction | `CCQEXSecCorr_SBN_v3_CCQEXSecCorr` |
| `GENIE_NCEL` | NC elastic | `GENIEReWeight_ICARUS_v2_multisigma_MaNCEL`<br>`GENIEReWeight_ICARUS_v2_multisigma_EtaNCEL` |
| `GENIE_RES` | GENIE resonance (CC + NC) + shared decay dials | `GENIEReWeight_ICARUS_v2_multisigma_M[av]CCRES`<br>`GENIEReWeight_ICARUS_v2_multisigma_M[av]NCRES`<br>`GENIEReWeight_ICARUS_v2_multisigma_RDecBR1eta`<br>`GENIEReWeight_ICARUS_v2_multisigma_RDecBR1gamma`<br>`GENIEReWeight_ICARUS_v2_multisigma_Theta_Delta2Npi` |
| `ResIso_RES` | ResIso resonance + the same shared decay dials | `ResIso_ResIso_ResonanceIsolation_MaRes_ResCode[0-1]`<br>`ResIso_ResIso_ResonanceIsolation_MvRes_ResCode[0-1]`<br>`GENIEReWeight_ICARUS_v2_multisigma_RDecBR1eta`<br>`GENIEReWeight_ICARUS_v2_multisigma_RDecBR1gamma`<br>`GENIEReWeight_ICARUS_v2_multisigma_Theta_Delta2Npi` |
| `GENIE_NonRES_CC` | Non-resonant CC 1π/2π | `GENIEReWeight_ICARUS_v2_multisigma_NonRESBG*CC[1-2]pi` |
| `GENIE_NonRES_NC` | Non-resonant NC 1π/2π | `GENIEReWeight_ICARUS_v2_multisigma_NonRESBG*NC[1-2]pi` |
| `GENIE_DIS_BY` | DIS / Bodek–Yang | `GENIEReWeight_ICARUS_v2_multisigma_[AB]htBY`<br>`GENIEReWeight_ICARUS_v2_multisigma_CV[1-2]uBY` |
| `MEC_EnergyDependence` | MEC energy dependence | `GENIEReWeight_ICARUS_v2_multisigma_EnergyDependence_CCMEC` |
| `MEC_ComponentFractions` | MEC Δ and pn fractions | `GENIEReWeight_ICARUS_v2_multisigma_FracDelta_CCMEC`<br>`GENIEReWeight_ICARUS_v2_multisigma_FracPN_CCMEC` |
| `MEC_XSecShape` | MEC cross-section shape | `GENIEReWeight_ICARUS_v2_multisigma_XSecShape_CCMEC`<br>`GENIEReWeight_ICARUS_v2_multisigma_XSecShape_CCMEC_Empirical`<br>`GENIEReWeight_ICARUS_v2_multisigma_XSecShape_CCMEC_Martini` |
| `MEC_DecayKinematics` | MEC decay angles | `GENIEReWeight_ICARUS_v2_multisigma_DecayAngMEC`<br>`GENIEReWeight_ICARUS_v2_multisigma_DecayAng2MEC`<br>`GENIEReWeight_ICARUS_v2_multisigma_DecayAngMECVariationResponse` |
| `MECTemplate_SuSAToMartini` | MEC template | `MECq0q3InterpWeighting_SBN_v3_SuSAToMar_MECResponse_q0bin[0-3]` |
| `MECTemplate_SuSAToValencia` | MEC template | `MECq0q3InterpWeighting_SBN_v3_SuSAToVal_MECResponse_q0bin[0-3]` |
| `MEC_C12ToAr40_Scaling` | 2p2h C → Ar scaling | `MiscInteractionSysts_DUNE_TDR_FakeData_C12ToAr40_2p2hScaling_nu`<br>`MiscInteractionSysts_DUNE_TDR_FakeData_C12ToAr40_2p2hScaling_nubar` |
| `FSI_Nucleon` | Nucleon FSI | `GENIEReWeight_ICARUS_v2_multisigma_FrAbs_N`<br>`GENIEReWeight_ICARUS_v2_multisigma_FrCEx_N`<br>`GENIEReWeight_ICARUS_v2_multisigma_FrInel_N`<br>`GENIEReWeight_ICARUS_v2_multisigma_FrPiProd_N`<br>`GENIEReWeight_ICARUS_v2_multisigma_MFP_N` |
| `FSI_Pion` | Pion FSI | `GENIEReWeight_ICARUS_v2_multisigma_FrAbs_pi`<br>`GENIEReWeight_ICARUS_v2_multisigma_FrCEx_pi`<br>`GENIEReWeight_ICARUS_v2_multisigma_FrInel_pi`<br>`GENIEReWeight_ICARUS_v2_multisigma_FrPiProd_pi`<br>`GENIEReWeight_ICARUS_v2_multisigma_MFP_pi` |
| `PionAbs_QuasiDeuteron` | Pion absorption, quasi-deuteron fraction | `PionAbsWeighter_SBN_v3_QuasiDeuteronFraction` |
| `WoodsSaxon` | Nuclear density (Woods–Saxon) | `WSReweight_v1_nucleus_radius`<br>`WSReweight_v1_surface_thickness` |
| `Flavor_NuNuBar_Ratios` | ν<sub>e</sub>/ν<sub>μ</sub> and ν/ν̄ cross-section ratios | `MiscInteractionSysts_DUNE_TDR_FakeData_nuenuebar_xsec_ratio`<br>`MiscInteractionSysts_DUNE_TDR_FakeData_nuenumu_xsec_ratio` |

* **Coverage:** every dial in the weight file is in at least one group, except `GENIEReWeight_ICARUS_v2_multisigma_Fakedial`.
* **Overlap:** `RDecBR1eta`, `RDecBR1gamma` and `Theta_Delta2Npi` are in both `GENIE_RES` and `ResIso_RES`, on purpose. Covariance matrices of these two groups therefore must **not** be added together, or those three dials count twice. For an overall total, run `syst_covariance.py --dials "*" --exclude "*Fakedial"`, which counts each dial once.
* **Running every group** (one output folder each):

```bash
for g in $(python -c "from plot_cv_weights import GROUPS; print(' '.join(GROUPS))"); do
    python syst_covariance.py --group "$g" --binning atm --variables reco_cosz \
        --samples numu nue --short-labels --outdir "cov_groups/$g"
done
```

---

## 9. Notes and known issues

* **ZExpPCA MvA dials:**
  * the four `MvA_b1..b4` dials have identical CV weights;
  * their CV weights have a long tail (up to about 75) and a few negative values.

  In the spectra this shows up as "dial off" ≈ 3–5 % of CV and spiky bins. It is worth checking with whoever produced the weights.
* **Base weight:** the default is unoscillated `xsec × flux_<true flavour>`. ν<sub>τ</sub> (or any |pdg| other than 12/14) gets base weight 0. The normalisation of `xsec` and `flux_*` has not been checked on the real CAF. `--base-weight final` uses `final_oscillated_w` instead.
* **b-scan grouping** is by name (`<provider>_b<N>`, two or more scans of one provider). Check the "shared zero response, applied once" line in the output lists the groups you expect.
* **Empty reco entries:** `systs.py` turns them into −999, which can pass its filter. Here they are removed by the quality cut.
* **Option names:** all scripts reject abbreviated or mistyped options (`allow_abbrev=False`) instead of guessing.
* **Testing:** all scripts (including `check_weights.py`) have been checked against independent calculations on mock ROOT files with the same tree and branch layout. They have **not** yet been validated on the real files listed in §2.
* **Particle types and `walldist` are not readable with uproot:** in `caf_new_sum.2.6M_weighted.root`, `common.ixn.pandora.part.pandora` is stored memberwise ("memberwise serialization of AsArray").
  * `make_part_info.py` reads them with ROOT (§6c). It was tested on a mock CAF written with the same nesting, where uproot fails in the same way.
  * Without `part_info.npz`, `--nparticles-from auto` uses `part.npandora` − 1, which assumes one unassociated-hits object per interaction. FC/PC is then unavailable.
* **`make_part_info.py` on the real file:** `MakeProject` must compile the full `caf::StandardRecord` from the file's streamer info. This needs a C++ compiler and `rootcling`, and was not run on the real CAF. If it fails, use `--dict-lib` with the duneanaobj dictionary from your DUNE software setup.
* **FC/PC definition** uses the minimum `walldist` over non-hit-collection particles of the first interaction. If your analysis defines containment differently (e.g. per track end point, or including the hit collection), the rule is in `partinfo_walldist_min` in `make_part_info.py`.

---

## 10. Changelog

| Date | Change |
|---|---|
| 2026-10-29 | `plot_cv_weights.py`: CV weight histograms; CV from `systsHeader/cv`; `--logx`/`--logy`; distinct colours; note on identical dials |
| 2026-10-29 | `plot_syst_spectra.py`: reco/true spectra per CVN sample at CV and ±nσ (definitions from `systs.py`); other dials at CV; ratio panels |
| 2026-10-29 | Custom binning: `--edges`, `--binning atm`, `--uniform-axis`, `--per-width`; `--logx` |
| 2026-10-29 | Ratio panel auto-range fits all curves; `--ratio-logy` |
| 2026-10-29 | `syst_covariance.py`: per-dial ±1σ universe covariance, summed over dials, correlation, fractional covariance, error band, `.npz`/`.root` output |
| 2026-10-29 | Covariance: ratio panel on the band, `--per-dial-plots`, physical `--logx`, ±σ universe curves on band plots (`--band-universes`) |
| 2026-10-29 | Default input paths set to the CAF and weight files in §2 |
| 2026-10-29 | README added |
| 2026-10-29 | Sample selection changed from CVN argmax to CVN thresholds (numu > 0.56, nue > 0.55; larger score wins when both pass); `--selection`, `--cvn-numu-cut`, `--cvn-nue-cut` |
| 2026-10-29 | Reco cos θ<sub>z</sub> = −`dir.calo.y` for all samples (quality cut now requires a non-zero calo direction); old estimator kept as `--cosz systs` |
| 2026-10-29 | `check_weights.py` added; `--base-weight xsec_osc\|osc\|final` option in spectra and covariance scripts |
| 2026-10-29 | Base weight now from the flux branches: default `flux_osc` = xsec × (flux_nue·osc_from_e_w + flux_numu·osc_from_mu_w); `flux_sample` picks the flux by reco CVN sample; `check_weights.py` tests the flux formulas |
| 2026-10-29 | Simplified: oscillated case only, base weight = `final_oscillated_w`; `--base-weight` and `--unoscillated` removed; `check_weights.py` kept as an optional diagnostic |
| 2026-10-29 | `plot_reco_cosz.py` added (from the user's heshw/lngtrk cos θ<sub>z</sub> script): y axis weighted by `final_oscillated_w` × dial CVs; `--syst`, `--cv-dials`, `--no-dial-cv`, `--unweighted`, `--selection mach3\|analysis`; default files |
| 2026-10-30 | Combined samples `nue+numu` and `all` (`--samples`) in spectra and covariance scripts |
| 2026-10-30 | `--detsys-compatible` (Claire's detsys conditions), `--cosz detsys`, `--selection mach3`; covariance band plot `--per-width` / `--ylabel` |
| 2026-10-30 | `--interaction all\|cc\|nc` (true `mc.nu.iscc`) in spectra and covariance scripts; `--cc-only` kept as a shortcut; choice shown in plot titles |
| 2026-10-30 | Default reco cos θ<sub>z</sub> = MaCh3 direction (`--cosz mach3`): `dir.calo` if < 2 pandora particles (hit collection excluded) or `Enu.calo` ≥ 1.3 GeV / NaN, else `heshw` (ν<sub>e</sub>-like) / `lngtrk`; normalised −d<sub>y</sub>/\|d\| |
| 2026-10-30 | Weights: default base = unoscillated xsec × true-flavour flux (`--base-weight flux_xsec`; `final` keeps `final_oscillated_w`); raw stored responses (negatives kept, nonfinite → missing bin; `--clip-responses` for the old handling); shared b-scan zero response applied once per provider (`--independent-b-scans`); NaN-safe histogramming; covariance marks missing bins. Applied in spectra, covariance and `plot_reco_cosz.py` |
| 2026-10-30 | `plot_reco_cosz.py`: ν<sub>e</sub>-like shows only `heshw`, ν<sub>μ</sub>-like only `lngtrk`; `--show-both` for the overlay |
| 2026-10-30 | Checked `--cosz mach3` against the MaCh3 C++ algorithm (line-by-line port: identical); `plot_reco_cosz.py` now uses it by default (`--cosz mach3`), raw heshw/lngtrk kept as `--cosz pfp` |
| 2026-10-30 | Real CAF stores `part.pandora` memberwise (unreadable by uproot): `--nparticles-from auto\|type\|npandora\|npfps`, auto falls back to `part.npandora` − 1 and prints the npandora/npfps agreement; same in `plot_reco_cosz.py` |
| 2026-10-30 | Dial groups replaced by the user's 27 physics groups (CCQE axial ×3, vector, templates, QE interference, NCEL, RES/ResIso, NonRES CC/NC, DIS, MEC ×8, FSI, pion absorption, Woods–Saxon, flavour ratios) |
| 2026-10-30 | FC/PC: `make_part_info.py` (PyROOT/RDataFrame, classes from `MakeProject`) → `part_info.npz` with `walldist_min` and exact nParticles; `--containment fc pc`, `--walldist-cut`, `--part-info`; new variables `reco_y`/`true_y`; default binning `atm_fcpc` with the FC/PC ν<sub>e</sub>/ν<sub>μ</sub> E<sub>ν</sub>, y and cos θ<sub>z</sub> edges |
| 2026-10-31 | `make_part_info.py`: if `rootcint`/`rootcling` is missing (pip ROOT), compile the MakeProject classes with ACLiC (only g++ needed) |
| 2026-11-01 | `plot_syst_spectra_josh.py` and `syst_covariance_josh.py`: Josh's truth-channel samples, common selection with cut flow, FC > 10 / PC 0–10 cm, hybrid direction (NC → calo), e_had energies, q0 truth, one-dial weights with nonfinite → 1, ±9999 clip, ≤ 0 → 0; band plots take a `cv_label` |
| 2026-11-01 | Code put in a git repository: `requirements.txt`, `.gitignore` |
| 2026-11-02 | `plot_syst_spectra_josh.py`: y axis is events / bin width by default ("Events / bin width"); `--no-per-width` for events per bin |
