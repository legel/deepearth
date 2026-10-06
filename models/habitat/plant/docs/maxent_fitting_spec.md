# How maxent.jar fits a model: a specification for reimplementation

> **Status (2026-10-04).** Implemented in `ranges/maxent_torch.py` and validated against maxent.jar
> on 34 species; the validation report is [maxent_torch.md](maxent_torch.md). Two findings came from that
> validation and are folded in below: §1.1 (layers are sorted alphabetically) and §7 (cross-validation folds).

Source: the open-source MaxEnt 3.4.x Java source (`density/` package, MIT licence),
read from `.../scratchpad/maxent_src/density/`. All `File.java:L` references are
to that tree. This document covers only the code paths used by our call:

```
nowarnings noprefixes -a -z autofeature=false linear=true quadratic=false
product=false threshold=true hinge=true betamultiplier=<b> responsecurves=false
jackknife=false pictures=false plots=false randomseed=false outputformat=cloglog
[final models: replicates=5 replicatetype=subsample randomtestpoints=25 threads=5]
```

The run is in SWD mode: `environmentallayers` is a CSV file, so `gridsFromFile()` is true
(`Runner.java:139`). All variables are continuous.

Notation: `m` is the number of training presences, `N` is the number of points the density is
defined over (background plus added presences, §1), `bm` is `betamultiplier`, and `x_ij` is
the raw value of variable `j` at point `i`.

---

## 0. Relevant parameter defaults (`parameters.csv`)

| parameter | default | effect for us |
|---|---|---|
| `removeDuplicates` | TRUE | per-species dedup by coordinates (§1.2) |
| `addSamplesToBackground` | TRUE | presences are appended to the background (§1.3) |
| `addAllSamplesToBackground` | FALSE | only presences whose environment vector is new are added |
| `doClamp` | TRUE | features are wrapped in `ClampedFeature` (a no-op on training data) and clamped at projection time |
| `extrapolate` | TRUE | no strict zeroing outside the training range |
| `cacheFeatures` | TRUE | `CachedScaledFeature` (numerically the same as `ScaledFeature`) |
| `maximumiterations` | 500 | `Sequential.maxIterations` |
| `convergenceThreshold` | 1e-5 | `Sequential.terminationThreshold` |
| `parallelUpdateFrequency` | **30** | overrides the `Sequential` field default of 10 (`Runner.java:1713`) |
| `beta_threshold/hinge/lqp/categorical` | -1 | automatic betas (§3) |
| `maximumbackground` | 10000 | **ignored in SWD mode**; it is used only by `Extractor` (`Runner.java:216`). The whole background file is used. |
| `allowpartialdata` | FALSE | presences with any NODATA value are dropped |
| `nodata` | -9999 | NODATA marker |
| `defaultprevalence` | 0.5 | logistic output only (not used with cloglog) |
| `threads` | 1 | affects only pictures, response curves and jackknife; **the optimiser is single-threaded** |

---

## 1. Data preparation

### 1.1 Reading
* **Background** (`GridSetFromFile.java:35-98`). Columns 4 onward are the variables
  (`DirectorySelect.getFiles`, `DirectorySelect.java:172-174`). **`params.layers` is then sorted
  alphabetically** (`ParamsPre.setSelections` → `getSelected`, `ParamsPre.java:148, 182-189`,
  `Arrays.sort`). This sorted order, not the file's column order, fixes the feature order, the
  `j % 20` refresh cycle (§5.1) and tie-breaking. Values are parsed as `double` and kept as `double`
  (`allvals`). Any row with a value exactly equal to NODATA (-9999) in any variable is
  **skipped** (`:73-78`). There is **no deduplication** and no subsampling.
* **Samples** (`SampleSet2.read`, `SampleSet2.java:108-195`). Species names are sanitised
  (spaces become `_`, and other characters are replaced, `SampleSet.java:121-135`). Values are
  parsed as `double`. A row with NODATA or an empty field is dropped (`:171-174`, because
  `allowpartialdata=false`). Presence feature values are read from `featureMap` as `Double`
  (`GridSetFromFile.java:124-128`). If the header suggests that column 2 is latitude, lon and
  lat are swapped (`:133`).

### 1.2 Duplicate removal (`Runner.java:321-322`, `SampleSet.java:137-153`)
Duplicate removal runs before any replicate or test split. In SWD mode `dim==null`, so the key
is the **string** `Double.toString(lat) + Double.toString(lon)`. The first occurrence in file
order is kept. Records with identical environment values but different coordinates are **not**
removed.

### 1.3 Adding presences to the background (`Runner.java:333-335, 434-440, 1220-1271`)
This is done separately for each run (each replicate), using **that run's training
presences** `ss` (after the test split):

1. `rnd[j] = Utils.generator.nextDouble()` is drawn for each variable (this consumes the RNG).
2. Each background point gets the hash `h_i = Σ_j rnd[j]·x_ij`, summed in variable order. The
   hashes are sorted.
3. A presence is added **iff** its hash (computed the same way) is **not found** by
   `Arrays.binarySearch`. In effect, a presence is added unless its environment vector is
   exactly equal (bitwise, as `double`) to some background row. Presences that duplicate
   each other in environment space are **all** added (there is no dedup among presences).
4. If nothing is added, the plain background is used (`:1252`).
5. Otherwise every base feature becomes a `FeatureWithSamplesAsPoints`
   (`FeatureWithSamplesAsPoints.java:36-38`). Points `0..N_bg-1` are background and points
   `N_bg..N-1` are the added presences, in `ss` order.

**Consequence:** the density is defined over `N = N_bg + N_added` points. Every training
presence's environment vector is present among the points. Test presences are **not** added.
`numBackgroundPoints` in `.lambdas` equals `N`.

### 1.4 Weights
No bias file is used, so `biasDist = ConstFeature(1)` and `biasDiv = ConstFeature(1)`
(`FeaturedSpace.java:352-366`). All points and presences have weight 1.
`bNumPoints() = N` (`:681-686`).

---

## 2. Features

### 2.1 Construction order (`Runner.makeFeatures`, `Runner.java:2145-2218`)
Let `cont[k]` be the variables in layer order. Because `doClamp=TRUE`, each one is wrapped as
`naturallyClamped(FWSAP)`, which clamps to [min, max] over the N points (`:2228-2238`). This
is a no-op on training data. The feature list is built in this order:

1. `LinearFeature(cont[k])` for every k (`:2176-2177`; always added, even if `linear=false`).
2. `ThrGeneratorFeature(cont[k])` for every k (`:2189-2191`).
3. For every k: `HingeGeneratorFeature(cont[k])`, then
   `HingeGeneratorFeature(revFeature(cont[k]))` named `<var>__rev` (`:2192-2196`).
   `revFeature` returns `-x` (`:2220-2226`).

Linear features are then wrapped (`:2204-2212`) as
`CachedScaledFeature(naturallyClamped(LinearFeature))`. Generator features are **not
scaled**: they work on raw values.

In `FeaturedSpace(...)` (`FeaturedSpace.java:304-325`), the generator features become
`ThrFeatureGenerator` or `HingeFeatureGenerator` objects in `featureGenerators[]`, keeping the
order above (all threshold generators, then hinge forward/reverse interleaved by variable).
The linear features form `features[]`.

### 2.2 Linear features (scaling)
`ScaledFeature(f)` (`ScaledFeature.java:38-54`) sets `min`/`max` over **all N points**
(background ∪ added presences, which equals background ∪ all training presences). The value is
`(x - min)/scale`, with `scale = max - min`; if `scale == 0`, then `scale = 1`. During training
the values lie in [0, 1].

### 2.3 Threshold and hinge candidate grid (`SortedFeatureGenerator.java:99-172`)
For each generator (variable `x`, or `-x` for the reverse hinge):

1. The generator's presences are the presences with data (all of them, for us), so `n_s = m`.
2. Build the combined list of `N` point values and `m` presence values. Sort it ascending with
   Java's **stable** `Arrays.sort`. Ties keep insertion order: points 0..N-1 first, then
   presences.
3. `prec = min over the combined list of precision(v)`, where `precision` is implemented at
   `:78-97`. `precision` returns roughly half a unit of the last significant decimal digit,
   with at most 6 significant digits. It looks for `00/01/99/98` digit pairs to detect
   floating-point noise. **`precision(0) = 0`** (log(0) = -inf, so `currentPower` = 0), so a
   single exact 0 anywhere makes `prec = 0`. Port it literally; it uses `Math.log`,
   `Math.pow`, `Math.floor` and `(int)` truncation.
4. `minVal`/`maxVal` are the first and last values of the sorted combined list.
5. Walk the sorted list with `lastVal = minVal`. At sorted index `i` with value `v`, if
   `v - lastVal > prec`, a new threshold starts:
   `thr[t] = (v + lastVal)/2`, `thrToVal[t] = i`, `valToThr[i] = t`, `lastVal = v`.
   Otherwise `valToThr[i] = -1`.
   **`lastVal` is the value at the start of the previous group, not its maximum.** When
   `prec = 0` this reduces to "midpoint between consecutive distinct values".
   `numThr` = (number of groups) - 1.
6. **Candidate index range.**
   * `ThrFeatureGenerator` (`ThrFeatureGenerator.java:34-48`): `thrFirst` is the first `t`
     with `minSampleValue <= thr[t]`, and `thrLast` is the count of `t` with
     `thr[t] < maxSampleValue`. Candidates are `t ∈ [thrFirst, thrLast)`, meaning thresholds
     that fall strictly inside the presence range.
   * `HingeFeatureGenerator` (`HingeFeatureGenerator.java:36-40`): `t ∈ [0, numThr)`, which
     is **every** threshold, including those above the largest presence.

Number of candidates per variable is about `(#thresholds inside the presence range) + 2·numThr`,
which is O(N) per variable.

### 2.4 Candidate feature functions
* Threshold: `T_t(x) = 1[x > thr[t]]` (`ThrFeatureGenerator.java:106-112`). This is binary
  (`isBinary()=true`, `ThresholdFeature.java:41`).
* Forward hinge: `H_t(x) = x > thr[t] ? (x - thr[t])/(maxVal - thr[t]) : 0`
  (`HingeFeatureGenerator.java:104-112`), with `maxVal` the generator's `maxVal` over
  points ∪ presences.
* Reverse hinge: the same function applied to `-x` with the reverse generator's
  `thr`/`maxVal`, where `maxVal_rev = -min(x)`.

### 2.5 Exporting a candidate as a concrete feature (`Sequential.java:91-98`)
When a generator candidate wins selection (§5.2) and does not yet exist, `exportFeature(t)`
creates a `ThresholdFeature(feature, thr[t])` or `HingeFeature(feature, thr[t], maxVal)` with
`isGenerated()=true`. It copies the generator's `sampleExpectation`, `sampleDeviation`,
`expectation` and `beta` (`ThrFeatureGenerator.java:114-128`,
`HingeFeatureGenerator.java:114-128`) and **appends** it to `X.features`. A generated feature
is never selected through the first loop of `getBestFeature`; it is always reconsidered
through its generator, which returns the existing object via `getFeature(t)`.

### 2.6 Names written to `.lambdas` (`FeaturedSpace.writeWeights`, `:750-781`)
* Linear: `<var>, λ, min, max` (ScaledFeature min/max in raw units). **Always written**, even
  when λ = 0.
* Threshold: `(<thr><<var>), λ, 0.0, 1.0`. The name is `"(" + thr + "<" + var + ")"` using
  Java `Double.toString` (`ThresholdFeature.java:33`).
* Forward hinge: `'<var>, λ, thr, maxVal`.
* Reverse hinge: the name `'<var>__rev` is rewritten to `` `<var>`` and written as
  `` `<var>, λ, -maxVal_rev, -thr_rev`` (that is, `min(x)` and the knot in x units).
  At projection it is evaluated as `HingeGrid(-x, -max, -min)`, which gives
  `x < knot ? (knot - x)/(knot - min) : 0` (`Project.java:193-199`).
* Non-linear features with λ = 0 are skipped. The order is linear features in layer order,
  then generated features **in the order they were first selected**.
* Trailer: `linearPredictorNormalizer`, `densityNormalizer`, `numBackgroundPoints`, `entropy`
  (§6).

---

## 3. Regularisation

### 3.1 Per-class β (`Runner.autoSetBeta`, `Runner.java:2253-2310`; `interpolate` `:2240-2250`)
`interpolate(x[], y[], n)`: find the first `i` with `n <= x[i]`. If `i == 0`, return `y[0]`.
If `i == len`, return `y[len-1]`. Otherwise return
`y[i-1] + (y[i]-y[i-1])·(n-x[i-1])/(x[i]-x[i-1])` in double arithmetic.
Here `n = ss.length`, the number of **training** presences for this run.

* `β_lqp`: because `product=false` and `quadratic=false`, this is the **linear-only table**
  `x={10,30,100}`, `y={1.0,0.2,0.05}` (`:2264-2267`). The table depends on which classes are
  enabled; with quadratic on, a different table would apply.
* `β_thr = interpolate({0,100},{2.0,1.0},n)`, which is `2 - n/100` for n<100 and 1 otherwise.
* `β_hinge = 0.5`.
* Each is multiplied by `bm`. Linear features get `β_lqp·bm`; generators get `β_thr·bm` or
  `β_hinge·bm` (`:2291-2299`), which is read by `SortedFeatureGenerator` as `beta`
  (`SortedFeatureGenerator.java:101`).

### 3.2 Sample mean and deviation (the "β_j" used in the penalty)
With no bias, `biasInfo = (avg=1, std=0, min=max=1)`, so dividing by the bias interval [1,1]
is the identity (`FeaturedSpace.java:397-424, 469-493`). For every feature:

```
low  = avg - beta/sqrt(cnt)*std ;  high = avg + beta/sqrt(cnt)*std
sampleExpectation N1 = 0.5*(low+high)          (= avg up to 1 ulp)
sampleDeviation   β_j = 0.5*(high-low)          (= beta*std/sqrt(cnt))
if β_j < minDeviation: β_j = minDeviation,   minDeviation = 0.001*bm   (FeaturedSpace.java:297-299, 480-487)
```

`std` is computed differently for each class:

* **Linear** (`getDividedSampleInfo`, `:429-465`), over the scaled values of the training
  presences: `avg = Σv/cnt` and `std = sqrt((Σv² - cnt·avg²)/(cnt-1))` (a one-pass formula;
  0 if the radicand is negative). Then cap: `std = min(std, 0.5·(max-min over points))`,
  where the cap is 0.5 for a non-constant scaled feature. If `cnt == 1`, `std = 0.5·(max-min)`.
  If `cnt == 0`, the feature is set inactive.
* **Threshold candidate t** (`ThrFeatureGenerator.java:50-90`). Let `c` be the number of
  presences at sorted index ≥ `thrToVal[t]`. Then `avg = c/m` and
  `std = sqrt((c - m·avg²)/(m-1))`, capped at 0.5. For m==1: `avg = c` and `std = 0.5`.
  **No `1/√m` floor.**
* **Hinge candidate t** (`HingeFeatureGenerator.java:42-85`), summed over presences at sorted
  index ≥ `thrToVal[t]` with `d = maxVal - thr`:
  `avg = Σ(v-thr)/d / m` and `csum2 = Σ(v-thr)²/d²`.
  For m>1: `std = sqrt((csum2 - m·avg²)/(m-1))`, then **cap 0.5, then floor `1/√m`**. The
  floor is applied last, so it wins when m<4. For m==1: `std = 0.5`. These are computed with
  the expanded sums `sum1 - thr·wsum1` and `sum2 - 2·sum1·thr + thr²·wsum2`. Port them
  literally for bit-level parity.

**Edge case from `prec` grouping.** Candidate statistics count "presence with sorted index ≥
group start", but the exported feature evaluates `x > thr`. They disagree only for values
inside `(lastVal, lastVal+prec]` that are > thr. When `prec = 0` (any exact 0 in the
variable) they always agree.

### 3.3 The penalty
`reg = Σ_j |λ_j|·β_j` over all features in `X.features` (`FeaturedSpace.getL1reg`, `:189-194`).
In sequential steps it is maintained incrementally (`Sequential.java:388`) and recomputed in
parallel steps (`:383`).

---

## 4. Objective (minimised)

Points `i = 1..N` (background plus added presences). Features `f_j(x)` as above.

```
lp_i   = Σ_j λ_j f_j(x_i)                          linearPredictor
lpn    = max_i lp_i                                linearPredictorNormalizer (recomputed after each update)
d_i    = exp(lp_i - lpn)                           density
Z      = Σ_i d_i                                   densityNormalizer
N1_j   = sampleExpectation_j                        (mean over training presences)
Loss   = -Σ_j λ_j N1_j + lpn + log Z  +  Σ_j β_j |λ_j|
       = -(1/m) Σ_s λ·f(x_s) + log Σ_i exp(λ·f(x_i)) + Σ_j β_j|λ_j|
```

Source: `FeaturedSpace.getLoss` `:185-187`, `getN1` `:200-205`, `setDensity` `:688-703`,
`Sequential.getLoss` `:62-64`. Initially λ=0, so Loss = log N.
**Regularised training gain = log N - Loss** (`Runner.java:1716`).
Model expectations are `E_q[f_j] = Σ_i d_i f_j(x_i)/Z`.

---

## 5. Optimisation (`Sequential.run`, `Sequential.java:497-531`)

```
newLoss = Loss(0)
for iteration = 0 .. 499:
    oldLoss = newLoss
    if iteration > 0 and iteration % 30 == 0:  newLoss = doParallelUpdate()
    else:  h = getBestFeature(); if h == null: break;  newLoss = doSequentialUpdate(h)
    if terminationTest(newLoss): break
```

### 5.1 Expectation caching (this affects which feature is selected)
* Each feature object holds `expectation`. `setDensity(toUpdate)` (`:688-703`) refreshes it
  **only for features in `toUpdate`**, and it always refreshes **all generator candidates**
  (`updateFeatureExpectations`, which also writes into exported generated features).
* Non-generated features (here, the linear features) outside `toUpdate` keep **stale**
  expectations. `getBestFeature` and `featuresToUpdate` use these stale values.
* `featuresToUpdate()` (`Sequential.java:396-417`) contains, in this order:
  * every active, non-generated feature `j` with `iteration < lastChange_j + 10` or
    `j % 20 == iteration % 20`;
  * the top 5 entries of `DoubleIndexSort.sort(dlb)`, ascending (most negative first). `dlb`
    is computed over **all** `X.features`, including generated and inactive features, using
    the current possibly stale expectations. An entry is added only if it is active,
    non-generated and not already in the list.

  `DoubleIndexSort` is a Bentley–McIlroy quicksort and is **not stable**. Port it literally
  if tie-breaking at the 5th slot matters.
* `doSequentialUpdate` refreshes `h`'s expectation exactly unless `h` is generated or was
  refreshed in the previous iteration (`:421-422`).

To match exactly, a reimplementation should keep a per-feature cached expectation and
refresh it only where maxent.jar does. Computing exact expectations every iteration is
simpler. It changes the selection path slightly but should converge to almost the same
optimum (§8).

### 5.2 Feature selection (`getBestFeature`, `:66-102`)
`bestLb = 1.0`. Comparisons use strict `<`, so ties go to the earliest candidate.

1. Loop over `X.features` in index order, using active, non-generated features only:
   `lb = deltaLossBound(f)`.
2. Loop over generators in order, and within each over `t = first..last-1`:
   `lb = deltaLossBound(candidate t)`. Each candidate uses the generator's `N1`, `β`,
   expectation `W1`, and `λ` (λ is 0 if not yet exported).
3. If a generator candidate wins and is not yet exported, export it (§2.5).

`deltaLossBound(h)` (`:325-342`). If inactive, it returns 0. Otherwise:
```
W1=E_q[h], W0=1-W1, N1=sampleExp, N0=1-N1, β=sampleDev, λ
α = goodAlpha(h); if α infinite → 0
bound = -N1·α + log(W0 + W1·e^α) + β(|λ+α| - |λ|);  NaN → 0
```
`goodAlpha(h)` (`:294-317`):
```
if W0<1e-6 or W1<1e-6: return 0
if N1-β > 1e-6 and (a1 = log((N1-β)W0/((N0+β)W1))) + λ > 0: return a1
elif N0-β > 1e-6 and (a2 = log((N1+β)W0/((N0-β)W1))) + λ < 0: return a2
else return -λ
```
This is the Bernoulli/[0,1] bound. It is applied to linear and hinge features too, because
they take values in [0,1] on the points.

### 5.3 Sequential step (`doSequentialUpdate`, `:419-465`)
```
refresh E[h] if needed (§5.1);  h.lastChange = iteration
toUpdate = featuresToUpdate()                 # computed BEFORE the step
dlb = deltaLossBound(h)
if h is binary (threshold):
    α = reduceAlpha(goodAlpha(h));  newLoss = increaseLambda(h, α, toUpdate)
else:
    α = reduceAlpha(newtonStep(h));  newLoss = increaseLambda(h, α, toUpdate)
    if newLoss - oldLoss > dlb:              # Newton step did worse than the bound
        increaseLambda(h, -α, [h])           # undo (also refreshes generators)
        α = reduceAlpha(searchAlpha(h, goodAlpha(h)))
        newLoss = increaseLambda(h, α, toUpdate)
```
* `newtonStep(h)` (`:223-239`): `var = Σ d_i h_i²/Z - E[h]²`. If `var < 1e-12`, return 0.
  `step = -deriv(h)/var`. If `(step+λ)·λ < 0`, set `step = -λ`, so λ is never allowed to
  change sign.
* `deriv(h)` (`:144-160`): `g = E_q[h] - N1`. If λ>0, return `g+β`. If λ<0, return `g-β`.
  If λ=0: if `g+β>0` return `g+β`; else if `g-β<0` return `g-β`; else return 0. This is
  literal; note it is not the minimum-norm subgradient.
* `reduceAlpha(α)` (`:467-472`): α/50 if iteration<10, α/10 if <20, α/3 if <50, else α.
* `searchAlpha(h, α0)` (`:104-121`): `L(α) = -α·N1 + log Σ d_i e^{α h_i} + (|λ+α|-|λ|)·β`.
  Starting at α0, keep multiplying by 4 while `L(4α) < L(α)` and finite. Then try `2α` once
  and keep it if it is better.
* `increaseLambda(h, α, toUpdate)` (`Sequential.java:387-394`,
  `FeaturedSpace.java:656-665`): `reg += (|λ+α|-|λ|)β`; `λ += α`; `lp += α·h`; recompute `lpn`
  as the max; then `setDensity(toUpdate)`.

### 5.4 Parallel step every 30 iterations (`doParallelUpdate`, `:245-292`)
```
for all j: u_j = (not binary and λ_j != 0) ? λ_j - previousLambda_j : 0 ; previousLambda_j = λ_j
```
This direction includes generated **hinge** features and excludes threshold features.
`previousLambda` starts at 0 for features exported later.
```
newtonStep(u) (:175-219): over features with u_j≠0 (refreshing their E):
    uTY = Σ d_i (F_i·u)/Z ; uTHu = Σ d_i (F_i·u)²/Z - uTY² ; if uTHu<1e-12 → 0
    step = -(deriv·u)/uTHu ; for j in order: if (step·u_j+λ_j)·λ_j<0: step = -λ_j/u_j
α_j = step·u_j ; if α_j != -λ_j and |α_j+λ_j| < 1e-6: α_j = -λ_j     (snap to zero)
lossWas = current Loss ; toUpdate = featuresToUpdate()
lossNow = increaseLambda(α, toUpdate)        # FeaturedSpace.increaseLambda(double[],..) :728-737, then setReg
if lossNow > lossWas: apply -α (undo)
```
No `reduceAlpha` is applied to the parallel step.

### 5.5 Termination (`terminationTest`, `:474-495`)
* At iteration 0: `previousLoss = newLoss`.
* The test runs only when `iteration % 20 == 0`: stop if `previousLoss - newLoss < 1e-5`
  (the loss drop over the last 20 iterations), else `previousLoss = newLoss`.
* Stop also if gain > 10000.
* Otherwise stop at 500 iterations.

### 5.6 Determinism
The optimiser is deterministic: there is no RNG and it is single-threaded. Floating-point sums
are sequential in point order. A GPU reduction changes summation order, which gives about
1e-12 relative differences. Those can flip near-ties in `getBestFeature`, so the selection
path may diverge after many iterations. Lambdas should still agree closely and predictions
very closely.

---

## 6. Output and the .lambdas trailer

After `alg.run()`, `removeBiasDistribution()` re-runs `setDensity` (no change). Then
(`Runner.java:455-479`, `FeaturedSpace.java:87-96, 776-779`):

* `linearPredictorNormalizer = lpn = max_i lp_i` over the N training points, at the final λ.
* `densityNormalizer = Z = Σ_i exp(lp_i - lpn)`.
* `numBackgroundPoints = N` (background plus added presences).
* `entropy H = -Σ_i p_i ln p_i`, where `p_i = d_i/Z` (terms with p=0 skipped).

**Projection** (`Project.java:137-322`) with `doClamp=TRUE`:
* Variables are read through `Grid.eval`, which returns a **float32** cast of the value
  (`GridSetFromFile.java:107-110`). Training used doubles.
* Linear: `clamp01((x-min)/(max-min))`. The clamp is applied after scaling (`ScaledGrid`).
* Forward hinge: `x<=thr ? 0 : (x-thr)/(max-thr)`, with output 1 if `x>max`
  (`HingeGrid`, `:459-465`). The reverse hinge is the same function on `-x` with
  (`-max`, `-min`).
* Threshold: `x >= thr ? 1 : 0`. **Note `>=` at projection versus `>` in training**
  (`Project.java:448` vs `ThresholdFeature.java:38`).
* `raw = exp(Σλf - lpn)/Z`, capped at 1. `cloglog = 1 - exp(-raw·e^H)`
  (`Project.java:274-281, 331-333`).
* If the result is NaN or Inf, it becomes 1 (if the sum exceeds lpn) or 0.

---

## 7. Randomness and replicates

* `Utils.generator = new Random(randomseed ? currentTimeMillis : 0)` (`Runner.java:303`).
* **With `replicates=5`, `replicatetype=subsample` and `randomseed=false`, maxent forces
  `randomseed=true`** ("so that replicates are not identical", `Runner.java:258-261`). The
  seed is then the wall clock, so **final-model replicate splits are not reproducible
  between runs**. A reimplementation can only match their distribution, not the exact split.
* Single runs with `randomtestpoints>0` and `randomseed=false` reset the RNG to
  `new Random(11111)` before the split (`:365`).
* The order of operations is: removeDuplicates → `replicate(5, bootstrap=false)`, which
  creates `<sp>_0..<sp>_4` as full copies (`SampleSet.java:201-214`) → `randomSample(25)`
  over `getNames()` (`:217-234`). `randomSample` processes the base species and each
  replicate **independently**. For each, it draws
  `toRemove = (int)(25·n/100.0) = floor(n/4)` test points by repeatedly taking
  `sel = (int)(nextDouble()·size)` and removing it, without replacement. So the test sets
  of different replicates are independent draws and may overlap; replicates are not folds.
  Only the `_k` names are fitted (`Runner.java:356-361`).
* **Cross-validation** (`replicatetype=crossvalidate`, which is how our pipeline selects β). `randomseed` is
  *not* forced here (the check at `Runner.java:258` excludes CV), so the generator is `Random(0)`, and
  `splitForCV` is its first consumer (`Runner.java:352`, `SampleSet.java:174-199`):
  * draw `rnd[k] = nextDouble()` for every presence k in file order (after duplicate removal);
  * `order = DoubleIndexSort.sort(rnd)`;
  * presence k goes to fold `order[k] % min(n, 5)`.

  Fold j trains on the other presences in file order and tests on its own, also in file order. The
  `(average)` row of `maxentResults.csv` is the mean of the four-decimal printed per-fold values
  (`getJackMean`), which is what β selection reads. These folds are deterministic and reproducible bit for bit.
* Training-time RNG use is limited to the hash coefficients in §1.3. These only decide exact
  equality, so the fit does not depend on the seed beyond the split.

---

## 8. What a GPU reimplementation must match

**Must match exactly** (otherwise the fitted model differs structurally):
1. Point set = background (NODATA rows dropped) + training presences not exactly equal to a
   background row; the presence set after coordinate-string dedup.
2. Linear scaling min/max over the N points; `β_lqp` from the **linear** table, `β_thr`,
   `β_hinge = 0.5`; × `bm`; `minDeviation = 0.001·bm`.
3. Per-class std formulas, including the 0.5 cap, the hinge `1/√m` floor and the m==1 rules.
4. Threshold grid: stable sort, `precision()`/`prec` grouping, `thr = (start_k + start_{k-1})/2`,
   `maxVal` over points ∪ presences, the threshold candidate range inside the presence range,
   and all hinge candidates.
5. The loss, including `lpn`, and the `goodAlpha`/`deltaLossBound` selection rule. Newton
   with sign clipping, `reduceAlpha` schedule, undo and `searchAlpha` fallback, parallel step
   every 30 iterations, and the termination rule (20-iteration window, 1e-5, 500 iterations).
6. `.lambdas` format and trailer semantics (§2.6, §6).

**May differ within tolerance:**
* Summation order (GPU reductions) and float64 versus Java double transcendental ulps.
  Keep **float64** on the GPU; float32 is not acceptable for `log Z` or for the expectation
  differences `W1 - N1`.
* Stale-expectation bookkeeping (§5.1) and `DoubleIndexSort` tie order. Ignoring them
  changes the selection path, so the selected knot set may differ by a few low-weight
  features. Predictions should stay highly correlated.
* Replicate test splits (they cannot match, §7).

**Suggested validation protocol:**
1. Same SWD inputs, `replicates=1` and `randomtestpoints=0`, which makes maxent.jar
   deterministic. Compare `N`, the linear min/max, the per-feature β_j for selected
   features, and the trailer (`lpn`, `Z`, `entropy`) at the jar's λ. Evaluating the jar's
   λ with our feature code must reproduce its trailer to about 1e-10 (unit test of the
   feature and density code).
2. Fit both and compare: regularised gain (target |Δ| < 1e-3), iteration count, the set of
   selected features (Jaccard index), λ of matched features, and the Pearson and Spearman
   correlation of cloglog over background points and a map window. Also compare
   training AUC.
3. For replicates, compare distributions: mean test AUC and mean prediction across 5
   replicates from each engine.
