# DUNE atmospheric (ANOA) cross-section systematics tools

Python scripts to study cross-section systematic dials for the DUNE
atmospheric-neutrino oscillation analysis:

| Script | What it does |
|---|---|
| `plot_cv_weights.py` | Histograms the per-event weight of a group of dials at their central value (or any σ) |
| `plot_syst_spectra.py` | Reco and true spectra for true CC ν<sub>e</sub> / CC ν<sub>μ</sub> / NC channels, FC/PC, at each dial's cv and ±nσ, with ratio panels; one dial at a time (§5) |
| `syst_covariance.py` | Per-dial covariance matrices from ±σ universes, their sum, correlation matrices and per-bin error bands, same conventions (§6) |
| `syst_lib.py`, `cov_lib.py` | Shared helpers (reading, dials, binning, plotting, covariance); not run directly |
| `make_part_info.py` | **Run once with ROOT.** Per-event closest particle wall distance (for FC/PC) and exact particle count from the memberwise `part.pandora` branch → `part_info.npz` (required by §5 and §6) |
| `plot_reco_cosz.py` | Standalone: reco cos θ<sub>z</sub> from the shower (`heshw`) and track (`lngtrk`) directions for CVN ν<sub>e</sub>-like / ν<sub>μ</sub>-like samples (§6a) |
| `check_weights.py` | Optional diagnostic: statistics of the CAF `weights` branches and how `final_oscillated_w` relates to flux / xsec / oscillation branches |

All scripts must sit in the **same folder**: dial groups come from
`plot_cv_weights.py`, the spectra script uses `syst_lib.py`, and
`syst_covariance.py` uses `plot_syst_spectra.py`, `syst_lib.py` and
`cov_lib.py`. The selection, weights and binning are therefore identical
across spectra and covariance.

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

The defaults are `DEFAULT_CAF` / `DEFAULT_WEIGHTS` in `syst_lib.py` (for the
spectra and covariance scripts) and near the top of the other scripts. The CAF and weight files must match **entry by entry**: same number of
events, same order.

**Weight layout.** For a dial with `variations = [0, 1, -1, 2, -2, 3, -3]`,
the `SystWeights` branch of that dial holds, per event, a vector of 7 weights
in that same order. The script finds the weight at a given σ by looking up the
index of σ in `variations`. The central value is the entry at index
`variations.index(cv)`, with `cv` taken from `systsHeader/cv`. Correction dials
(0 or 1 entries in `variations`) use element 0.

## 3. Quick start

```bash
# once, with a Python that has PyROOT:
python make_part_info.py                 # -> part_info.npz (FC/PC + exact nParticles)
```

```bash
# Inspect the dials: cv, variations, which index is the CV
python plot_cv_weights.py --list

# 1) CV weight distributions for a group of dials
python plot_cv_weights.py --group CCQE_Axial_MvA_PCA --nbins 50 --logy --legend-stats --short-labels

# 2) Spectra at cv and +/-1,2,3 sigma, FC/PC, events / bin width
python plot_syst_spectra.py --group FSI_Pion --containment fc pc \
    --samples cc_nue cc_numu --variables reco_E reco_y reco_cosz --uniform-axis --short-labels

# 3) Covariance matrices + error band from +/-1 sigma universes
python syst_covariance.py --group FSI_Pion --containment fc pc \
    --samples cc_nue cc_numu --variables reco_E reco_y reco_cosz --per-width --per-dial-plots --short-labels
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

For every selected dial × containment × sample × variable it draws one figure:

* **Top panel:** the dial at its cv (black), +nσ (solid) and −nσ (dashed), one colour per n. Optional extras:
  * the nominal, no-dial curve (`--show-nominal`, dotted);
  * the MC-statistical band of the cv curve (`--stat-band`).
* **Bottom panel:** ratio to the cv curve.
* **Y axis:** events / bin width by default ("Events / bin width"). `--no-per-width` gives events per bin. The ratio panel is the same either way.

Output: `<outdir>/<[fc_|pc_]sample>/<variable>/<dial>.png`, default `<outdir>` = `syst_spectra/`.

`part_info.npz` from `make_part_info.py` is **required** (§6c): the containment
cut and the particle count come from it.

```bash
python make_part_info.py                       # once, with ROOT
python plot_syst_spectra.py --group CCQE_Axial_Deut_PCA --containment fc pc \
    --samples cc_nue cc_numu --variables reco_E reco_y reco_cosz --uniform-axis --show-nominal
python plot_syst_spectra.py --dials "*MaCCRES" --sigmas 1 --samples nc --variables reco_E true_E
```

### 5.1 Samples

Samples are **true interaction channels**, not CVN selections. ν and ν̄ are
together.

| `--samples` | Events |
|---|---|
| `cc_nue` | true CC, \|pdg\| = 12 (ν<sub>e</sub> and ν̄<sub>e</sub>), "electron channel" |
| `cc_numu` | true CC, \|pdg\| = 14 (ν<sub>μ</sub> and ν̄<sub>μ</sub>), "muon channel" |
| `nc` | true NC, \|pdg\| = 12 or 14 |
| `cc_nue+cc_numu`, `all` | sums of the above (`all` = CC ν<sub>e</sub> + CC ν<sub>μ</sub> + NC) |

ν<sub>τ</sub> is not a supported channel (it has no flux). Defaults: `cc_nue cc_numu nc`.

### 5.2 Common selection

Applied in order; the script prints the cut flow (events remaining after each
cut) and the events and nominal sum per sample. `--skip-cuts` drops any of them.

| Cut | Keeps events with |
|---|---|
| `one_nu` | exactly one MC neutrino in the entry |
| `one_ixn` | exactly one pandora interaction |
| `channel` | a supported truth channel (§5.1) |
| `base` | nominal weight (§5.4) finite and > 0 |
| `energy` | true E in [`--emin`, `--emax`] = [0.1, 100] GeV |
| `fv` | vertex inside the `--fv XMIN XMAX YMIN YMAX ZMIN ZMAX` box [cm] (`--fv-vertex true\|reco`); **off unless `--fv` is given** |
| `containment` | a valid containment minimum: 0 ≤ `walldist_min` ≤ `--walldist-max` (10000 cm) |
| `true_dir` | a valid true direction (\|p\| > 0) |
| `reco_dir` | a valid reco direction (finite, \|d\| > 0; horizontal is valid) |

Each observable also has its own finite-value mask. The script prints, per
sample and variable, how many events are finite and how many fall below or
above the bin edges (e.g. reco y outside [0, 1] stays in the data but outside
the histogram).

**FC / PC** (`--containment fc pc`, default `all`): FC is
`walldist_min` > `--walldist-cut` (10 cm); PC is 0 ≤ `walldist_min` ≤ 10 cm.
Missing, negative or > 10000 cm values are unknown and removed by the
containment cut. See §6c for how `walldist_min` is made.

### 5.3 Reconstruction and truth

| Quantity | Definition |
|---|---|
| nParticles | pandora particles of the interaction, excluding the hit collection (`origRecoObjType` = 3), from `part_info.npz` |
| Reco direction d | `dir.calo` if NC, or nParticles < 2, or !(`Enu.calo` < `--hits-ecut` = 1.3 GeV); otherwise `dir.heshw` (electron channel) or `dir.lngtrk` (muon channel) |
| `reco_cosz` | −d<sub>y</sub>/\|d\| (negative = upgoing) |
| `reco_E` | `Enu.e_calo` (electron channel), `Enu.lep_calo` (muon channel) |
| `reco_Ehad` | `Enu.e_had` for both channels |
| `reco_Elep` | `reco_E − reco_Ehad` |
| `reco_y` | `reco_Ehad / reco_E` if `reco_E` > 0, else 0 (values outside [0, 1] kept) |
| NC | `reco_E` = `reco_Ehad` = `Enu.calo` (visible energy); `reco_Elep` and `reco_y` are NaN |
| `--reco-energy matched_range` | muon channel uses `Enu.mu_range` / `Enu.mu_had`, valid only when both are finite and `mu_range` > 0 (default `pierre` = the rows above) |
| `true_E` | `mc.nu.E` |
| `true_Ehad` | q<sub>0</sub> |
| `true_Elep` | E − q<sub>0</sub> |
| `true_y` | `mc.nu.inelasticity` |
| `true_cosz` | −p<sub>y</sub>/\|p\| of the neutrino |

### 5.4 Weights (one dial at a time)

```
nominal  = xsec × flux_nue  (|pdg| = 12)  or  xsec × flux_numu  (|pdg| = 14)
           unoscillated, from the CAF 'weights' tree (newest cycle), no other factor
response = stored dial weight; nonfinite → 1; clipped to [−9999, 9999]
weight   = nominal × response; weights ≤ 0 are set to 0
```

Only the varied dial is applied: no other dial's CV, and no shared CV factors.
The black curve is the dial at its cv; `--show-nominal` adds the no-dial
nominal. An event with no entry at the requested variation gets response 1.

### 5.5 Main options

| Option | Meaning |
|---|---|
| `--group / --dials / --regex / --exclude`, `--list` | Dial selection (groups in §8) / list dials with cv and variations |
| `--sigmas 1 2 3`, `--only-up`, `--only-down` | Which shifts to draw; `--sigmas` with no values = cv only |
| `--show-nominal`, `--stat-band` | Extra curves (above) |
| `--samples`, `--containment`, `--variables` | What to plot (§5.1, §5.2; variables: `reco_`/`true_` `E`, `Elep`, `Ehad`, `cosz`, `y`) |
| `--walldist-cut 10`, `--walldist-max 10000`, `--part-info part_info.npz` | FC/PC definition and its input |
| `--emin`, `--emax`, `--fv`, `--fv-vertex`, `--skip-cuts` | Common selection (§5.2) |
| `--reco-energy pierre\|matched_range`, `--hits-ecut 1.3` | Reconstruction (§5.3) |
| `--binning`, `--edges`, `--bins`, `--nbins` | Binning (§7) |
| `--per-width` (default) / `--no-per-width`, `--uniform-axis`, `--logx` | y axis and x axis display |
| `--ratio-range lo hi`, `--ratio-logy`, `--logy` | Ratio panel range / log scales (auto range fits all curves) |
| `--title`, `--ylabel`, `--short-labels`, legend/font options | Labels |
| `--outdir syst_spectra`, `--format`, `--dpi` | Output |

---

## 6. `syst_covariance.py`: covariance matrices and error bands

Same samples, selection, reconstruction, weights and binning as §5 (it
imports them from `plot_syst_spectra.py`). `part_info.npz` is required.

### 6.1 Method

N is the vector of binned event counts. Every dial d is evaluated **on its
own** on top of the nominal weight (§5.4). Its universes are the dial at
cv − s and cv + s (default s = 1), compared with the dial's own cv
histogram ref<sub>d</sub>:

$$E^{(d)}_{ij} = \frac{1}{N_{\rm univ}}\sum_{\alpha}\left(N_i^{\alpha}-N_i^{{\rm ref},d}\right)\left(N_j^{\alpha}-N_j^{{\rm ref},d}\right)$$

The dials are treated as independent and **added**:

$$E_{ij} = \sum_{d} E^{(d)}_{ij} \;\;(+\ \text{optional statistical diagonal})$$

The **error band** in each bin is σ<sub>i</sub> = √E<sub>ii</sub>, drawn
around the nominal (no dial); the ratio panel shows 1 ± σ<sub>i</sub>/N<sub>i</sub>.
Per-dial band plots (`--per-dial-plots`) are drawn around that dial's ref.

Notes:
* Correction dials (one variation) have no universes and are skipped.
* With ±1σ universes, a dial's band edges coincide with its ±1σ spectra, which is a quick check of the construction.
* `--sigmas 1 2 3` also averages the ±2σ and ±3σ universes in with 1/N<sub>univ</sub>. The result is then no longer a 1σ covariance, and a warning is printed.
* Matrices are built from **event counts**, not a cross section.

### 6.2 Usage

```bash
# one matrix per (containment, sample, variable)
python syst_covariance.py --group CCQE_Axial_Deut_PCA --containment fc pc \
    --samples cc_nue cc_numu --variables reco_E reco_y reco_cosz --per-width --per-dial-plots

# one joint matrix (cross-sample, cross-variable and FC-PC correlations), + MC stat
python syst_covariance.py --group CCQE_Axial_Deut_PCA --containment fc pc \
    --samples cc_nue cc_numu --variables reco_E reco_y reco_cosz \
    --joint --add-stat mc --per-width --outdir syst_cov/joint
```

| Option | Meaning |
|---|---|
| `--sigmas 1` | Universes per dial: cv ± s for each s |
| `--add-stat none\|mc\|poisson` | Add Σw² (MC stat) or N (Poisson) on the diagonal |
| `--joint` | One matrix over all (containment, sample, variable) blocks instead of one per block |
| `--per-dial-plots` | Also plot each dial's own covariance, correlation and band in `<outdir>/dials/` |
| `--per-width`, `--ylabel` | Band plot divided by bin width, label "Events / bin width" (matrices unchanged; off by default here) |
| `--band-universes 5` | On the total band plot, overlay the ±σ spectra of the N largest dials (0 = none) |
| `--logx`, `--logx-zero x` | Physical log x (and log matrix axes) for energy variables, so bins keep their true relative widths. A lower edge of 0 is drawn one log-width below the next edge and still labelled "0" |
| `--top-dials 8` | Dials shown individually in the uncertainty breakdown |
| `--annotate`, `--cmap`, `--short-labels`, font/legend options | Style |
| `--outdir syst_cov`, `--format`, `--no-root` | Output (`--no-root`: only `.npz`) |
| Selection, reconstruction and binning options | Same as §5.5 |

### 6.3 Outputs (per matrix, `<outdir>/<[fc_|pc_]sample>_<variable>_*` or `joint_*`)

| File | Content |
|---|---|
| `*_cov.png` | Total covariance E<sub>ij</sub> |
| `*_fraccov.png` | Fractional covariance E<sub>ij</sub>/(N<sub>i</sub>N<sub>j</sub>) |
| `*_corr.png` | Correlation E<sub>ij</sub>/√(E<sub>ii</sub>E<sub>jj</sub>) |
| `*_fracerr.png` | √E<sub>ii</sub>/N<sub>i</sub> per bin: total and the largest dials |
| `*_band.png` | nominal ± √E<sub>ii</sub> with ±σ universe curves and a ratio panel |
| `*.npz` | `cv` (the nominal), `cov`, `cov_syst`, `cov_stat`, `frac_cov`, `corr`, `edges__<sample>__<var>`, `cov_dial__<dial>`, `universes__<dial>`, `universe_sigmas__<dial>`, `blocks`, `block_offsets` |
| `*_refs.npz` | `ref__<dial>`: each dial's cv reference histogram ref<sub>d</sub> |
| `*.root` | `cv`, `cov`, `cov_syst`, `frac_cov`, `corr` (TH1/TH2 on a bin-index axis), `dials/cov_<dial>`, `blocks/cv_<sample>_<var>` (physical edges) |

The script prints the matrix size, the total, the minimum eigenvalue, the
rank (each dial adds at most 2 with ±1σ) and the symmetry check for every
matrix.

---

## 6a. `plot_reco_cosz.py`: shower vs track reco cos θ<sub>z</sub>

Standalone (it does not need the other scripts) and keeps its **own**
conventions: CVN-selected samples and all dials at their CV. For each sample
(ν<sub>e</sub>-like blue, ν<sub>μ</sub>-like orange) it plots reco
cos θ<sub>z</sub> of the first pandora interaction.

* **`--cosz mach3` (default):** the MaCh3 algorithm. The direction is `dir.calo` if there are fewer than 2 particles (hit collection excluded) or `Enu.calo` ≥ 1.3 GeV or NaN. Otherwise it is `dir.heshw` for ν<sub>e</sub>-like and `dir.lngtrk` for ν<sub>μ</sub>-like, giving cos θ<sub>z</sub> = −d<sub>y</sub>/\|d\|. The legend gives the fraction of events that used `dir.calo`.
* **`--cosz pfp`:** the raw −`dir.heshw.y` (ν<sub>e</sub>-like) / −`dir.lngtrk.y` (ν<sub>μ</sub>-like), with no `dir.calo` switch and no normalisation. `--show-both` also overlays the other estimator (dashed).

With `--cvn`, "ν<sub>e</sub>-like" follows `--selection` (default `mach3`, the MaCh3 rule), as MaCh3's `isELike` does. Values outside [−1, 1] (CAF sentinels) and NaN are dropped.

**Weights (y axis):**

```
base = xsec × flux of the true flavour (|pdg| 12 → flux_nue, 14 → flux_numu, else 0), unoscillated
w    = base × Π_g w_g(cv)     one CV factor per dial; the <provider>_b1..bN scans share one zero response, applied once
```

Dial responses are used raw: negatives are kept, and a nonfinite response
makes only the bin that event falls in missing.

* `--cv-dials` restricts which dials enter the product (globs, default all).
* `--base-weight final` uses `final_oscillated_w` as the base; `--clip-responses` sets negative responses to 0 and nonfinite to 1; `--independent-b-scans` multiplies every b-scan.
* `--no-dial-cv` uses the base weight only; `--unweighted` gives plain counts.
* A bin touched by a nonfinite response is drawn as a gap, and the legend notes the missing bins.
* The legend shows N and Σw per curve, and the terminal prints Σw per sample, including NC-like.

**Samples:**
* `--cvn FILE... --syst FILE...`: classify with CVN; one systematics file per CAF file, same order.
  * `--selection mach3` (default): ν<sub>μ</sub> if `cvn.numu > 0.56`, else ν<sub>e</sub> if `cvn.nue > 0.55`, else NC.
  * `--selection analysis`: the same thresholds, but the larger score wins when both pass.
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

| | Rule (`--walldist-cut 10`, `--walldist-max 10000`) |
|---|---|
| FC | `walldist_min > 10 cm` |
| PC | `0 ≤ walldist_min ≤ 10 cm` (exactly 10 → PC) |
| unknown | missing (no particle with a defined `walldist`), negative or > 10000 cm: removed by the `containment` cut, also from `all` |

**Running.** `--containment fc pc` (or `all`, the default) in
`plot_syst_spectra.py` and `syst_covariance.py`.
* Each value gives its own plots or matrices: `fc_cc_numu/`, `pc_cc_nue_reco_E_*`, ….
* Titles start with "FC" or "PC".
* With `--joint`, all containments, samples and variables go into one matrix, which includes the FC–PC correlations.
* The scripts print the selected events per containment.

**Why a separate ROOT step.** `common.ixn.pandora.part.pandora` is stored
memberwise, which uproot cannot read. `make_part_info.py` reads it with PyROOT
+ RDataFrame:
* it generates the `caf::` classes from the CAF's own streamer info (`TFile::MakeProject`), so duneanaobj is not needed; `--dict-lib` uses an existing dictionary instead;
* it compiles them once into `caf_classes/` and reuses them afterwards. If ROOT's own build script fails because `rootcling`/`rootcint` is missing (as with the pip `ROOT` package), it compiles with ACLiC instead, which needs only `g++`. The `rootcint: command not found` / `ProjectDict.cxx` errors printed before that fallback are expected;
* it writes `part_info.npz` with `walldist_min`, `n_particles` and `n_entries`;
* `plot_syst_spectra.py` and `syst_covariance.py` read `--part-info part_info.npz` (the default), **stop if it is missing**, and check that it was made from a CAF with the same number of entries;
* the same file also gives the exact type-based nParticles used for the reco direction (§5.3).

```bash
python make_part_info.py --threads 8                     # needs PyROOT + a C++ compiler (first run only)
python plot_syst_spectra.py --group CCQE_Axial_Deut_PCA --containment fc pc \
    --samples cc_nue cc_numu --variables reco_E reco_y reco_cosz --uniform-axis
python syst_covariance.py --group CCQE_Axial_Deut_PCA --containment fc pc \
    --samples cc_nue cc_numu --variables reco_E reco_y reco_cosz --logx
```

---

## 7. Binning (spectra and covariance)

Priority, later wins: default uniform → `--nbins` → `--binning` preset →
`--bins var=n,lo,hi` → `--edges var=e0,e1,...`.

**`--binning atm_fcpc` (default).** It starts from the `atm` edges below. For
the FC and PC `cc_nue` / `cc_numu` samples it then uses these
sample-specific edges for E<sub>ν</sub>, y and cos θ<sub>z</sub> (reco and
true):

| Sample | E<sub>ν</sub> [GeV] | y | cos θ<sub>z</sub> |
|---|---|---|---|
| FC CC ν<sub>e</sub> | 0, 0.2, 0.4, 0.6, 0.8, 1, 1.1, 1.2, 1.35, 1.5, 1.7, 1.95, 2.25, 3.25, 100 | 0, 0.1, 0.2, 0.3, 0.4, 0.5, 1 | −1 … 1 in steps of 0.2 |
| FC CC ν<sub>μ</sub> | 0, 0.4, 0.6, 0.8, 1, 1.1, 1.2, 1.45, 1.65, 1.9, 100 | 0, 0.1, 0.2, 0.3, 0.4, 0.5, 1 | −1 … 1 in steps of 0.2 |
| PC CC ν<sub>e</sub> | 0, 0.2, 0.4, 0.6, 0.8, 1.05, 1.4, 2, 3.25, 100 | 0, 0.1, 0.2, 0.3, 0.4, 0.5, 1 | −1 … 1 in steps of 0.2 |
| PC CC ν<sub>μ</sub> | 0, 0.4, 0.6, 0.8, 1, 1.15, 1.3, 1.45, 1.6, 1.8, 2, 2.2, 2.45, 2.7, 3, 3.35, 3.75, 4.25, 4.85, 6.85, 100 | 0, 0.1, 0.2, 0.3, 1 | −1 … 1 in steps of 0.2 |

* **Other cases** use the general edges: containment `all`, combined samples, NC, and E<sub>lep</sub>/E<sub>had</sub> (which use `atm`). E<sub>ν</sub> and y otherwise get uniform bins.
* **Explicit binning wins:** a variable given with `--edges`/`--bins` overrides this table.
* **Where they live:** the edges are `FCPC_EDGES` in `syst_lib.py`.
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
| `--per-width` | Events / bin width (default in the spectra script; `--no-per-width` turns it off) |
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
    python syst_covariance.py --group "$g" --containment fc pc \
        --samples cc_nue cc_numu --variables reco_E reco_y reco_cosz \
        --per-width --short-labels --outdir "syst_cov/$g"
done
```

---

## 9. Notes and known issues

* **One-dial weights:** spectra and covariance apply only the varied dial on top of the nominal. Other dials' CVs (and corrections such as the ZExpPCA b-scan zero response) are **not** included, so the cv curve of a correction-heavy dial can differ visibly from the nominal.
* **Nominal weight:** unoscillated `xsec × flux_<true flavour>`. ν<sub>τ</sub> (or any |pdg| other than 12/14) is not a supported channel. The normalisation of `xsec` and `flux_*` has not been checked on the real CAF.
* **Response policy:** nonfinite responses become 1 and responses are clipped to ±9999; a total weight ≤ 0 becomes 0. So a broken knot never removes a bin, but it also gives no warning in the plot.
* **ZExpPCA MvA dials:** the four `MvA_b1..b4` dials have identical CV weights, with a long tail (up to about 75) and a few negative values. It is worth checking with whoever produced the weights.
* **Fiducial volume:** off unless `--fv` is given; the analysis box is not yet known.
* **Not implemented:** a signed angular 68 % width observable.
* **`plot_reco_cosz.py`** keeps its own CVN-sample, all-dials-at-CV conventions (§6a); it does not follow §5.
* **Option names:** all scripts reject abbreviated or mistyped options (`allow_abbrev=False`) instead of guessing.
* **Testing:** all scripts have been checked against independent calculations on mock ROOT files with the same tree and branch layout. They have **not** yet been validated on the real files listed in §2.
* **Particle types and `walldist` are not readable with uproot:** in `caf_new_sum.2.6M_weighted.root`, `common.ixn.pandora.part.pandora` is stored memberwise ("memberwise serialization of AsArray"). `make_part_info.py` reads them with ROOT (§6c); it was tested on a mock CAF written with the same nesting, where uproot fails in the same way.
* **`make_part_info.py` on the real file:** `MakeProject` must compile the full `caf::StandardRecord` from the file's streamer info. This needs a C++ compiler (and `rootcling`, or ACLiC as fallback). If it fails, use `--dict-lib` with the duneanaobj dictionary from your DUNE software setup.
* **FC/PC definition** uses the minimum `walldist` over non-hit-collection particles of the first interaction. If your analysis defines containment differently (e.g. per track end point, or including the hit collection), the rule is in `partinfo_walldist_min` in `make_part_info.py`.
