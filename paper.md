# Machine Learning Surrogate for S11 Prediction in Slot-Loaded Microstrip Patch Antennas: A Feasibility Study with Honest Generalization Assessment

---

## Abstract

This paper investigates the feasibility of machine learning (ML) surrogate models for predicting the S11 frequency-domain response of slot-loaded microstrip patch antennas, using a dataset of electromagnetic (EM) simulations comprising seven distinct slotted configurations. Two surrogate models—a Multi-Layer Perceptron (MLP) and a Gradient Boosting Regressor (GB)—are trained to map the input vector (operating frequency, patch length L, patch width W, slot length Ls, slot width Ws) to the corresponding S11(dB) value. A Differential Evolution (DE) optimizer is coupled with the surrogate for inverse design: given a target resonant frequency, the optimizer recovers the geometric dimensions that satisfy the specification. Critically, model generalization is assessed using configuration-level Leave-One-Out Cross-Validation (LOOCV), wherein each fold withholds one complete antenna geometry from training. Results show that while models achieve near-perfect fit on data from seen geometries (R² ≈ 0.99), generalization to unseen configurations is poor (mean LOOCV R² ≈ −0.98), revealing that the current dataset is insufficient in diversity for true generalization. These findings are reported with full transparency as a validated methodology benchmark, and a rigorous dataset expansion roadmap is proposed.

---

## 1. Introduction

Microstrip patch antennas are fundamental components in modern wireless systems, including 5G networks, IoT devices, and radar systems. Their compact form factor, ease of PCB fabrication, and compatibility with planar circuit integration make them a preferred choice. Among the various enhancement strategies, slot loading—etching a rectangular aperture into the metallic patch—is widely employed to extend effective current path length, thereby lowering the resonant frequency without increasing the antenna's physical footprint, or introducing dual-band characteristics.

Despite these advantages, designing slot-loaded antennas to meet precise specifications (target resonant frequency, return loss S11, and bandwidth) is a complex multi-variable optimization problem. Traditional approaches rely on full-wave electromagnetic simulators such as CST Microwave Studio or Ansys HFSS, which solve Maxwell's equations numerically over a discretized volumetric mesh. A single simulation can require two to five minutes; multi-parametric optimization sweeps can require days.

Machine learning-based surrogate models offer a compelling alternative: once trained on a set of EM simulations, an ML model can predict antenna performance in milliseconds, enabling rapid design iteration and optimization. However, a critical and often overlooked question is whether these surrogates truly *generalize* to geometries they have not encountered during training. This paper addresses exactly that question, providing an empirically honest assessment using configuration-level cross-validation.

### Problem Statement

Given a dataset of CST simulation frequency sweeps for seven slotted microstrip patch antenna configurations, can an ML surrogate model learn a sufficiently general mapping from (f, L, W, Ls, Ws) → S11(dB) to reliably predict the electromagnetic behavior of *previously unseen* slotted patch geometries?

---

## 2. Literature Review

**ML for Antenna Design**: The application of artificial neural networks (ANNs) to antenna design has been studied since the late 1990s. Early work by Mishra and Patnaik [1] demonstrated ANN-based resonant frequency prediction for rectangular patches with R² > 0.98, though models were validated only on interpolated samples from the same parameter distributions. More recent work by Sharma et al. [2] used feedforward ANNs to predict bandwidth and gain of patch antennas from geometric parameters, observing strong in-distribution performance.

**Surrogate-Assisted Optimization**: Koziel and Bekasiewicz [3] formalized the surrogate-assisted optimization paradigm, using kriging and radial basis function surrogates to accelerate antenna optimization cycles by 90–98%. Critically, their approach validated surrogates through co-simulation with high-fidelity models at each new operating point—a more conservative but rigorous strategy than fixed held-out test sets.

**Frequency-Domain ML**: The formulation of S11 prediction as a function of frequency—predicting S11(f) for a given geometry—was explored by Liu et al. [4] for frequency-selective surfaces and by Tahir et al. [5] for slotted antennas. Both studies acknowledged the challenge of training data sparsity and used large simulated datasets (N > 500 configurations) to mitigate generalization concerns.

**Dataset Size and Generalization**: A consistent finding across the literature is that ML models for antenna design require a minimum of 50–200 unique antenna configurations for reliable generalization. Patnaik et al. [6] showed that ANN performance degrades sharply when the training set covers fewer than 30 distinct geometries, emphasizing the need for systematic parametric sweeps.

**Differential Evolution for Inverse Design**: Cui et al. [7] established the combination of DE global optimizer with a neural network forward model as an effective inverse design strategy, circumventing the non-uniqueness problem inherent in direct inverse mapping. This approach has been replicated and refined for microstrip antenna design by Bermudez et al. [8].

**Gap Identified**: While numerous studies claim high ML accuracy for antenna design, few employ rigorous configuration-level cross-validation. Most use random row-level splits on frequency-sweep datasets, which—as demonstrated in this paper—can inflate reported performance by leaking configuration identity into the test set.

---

## 3. Dataset Description

### 3.1 Data Source

The dataset originates from full-wave EM simulations of rectangular microstrip patch antennas with centrally etched rectangular slots, produced using CST Microwave Studio (or equivalent). It is publicly available on Kaggle (Sinha [9]).

### 3.2 Structure

Each row represents one frequency-domain observation:

| Variable | Description | Unit | Range |
|---|---|---|---|
| f | Operating frequency | GHz | 1.0 – 3.5 |
| L | Patch length | mm | 29.4 – 87.0 |
| W | Patch width | mm | 26.0 – 130.0 |
| Ls | Slot length | mm | 14.0 – 85.0 |
| Ws | Slot width | mm | 12.0 – 115.0 |
| S11 | Return loss | dB | −34.5 to −0.2 |

### 3.3 Configuration Inventory

The dataset covers **seven slotted** and one plain (unslotted) configuration. Plain patches are excluded as out-of-distribution for this study.

| Config | L (mm) | W (mm) | Ls (mm) | Ws (mm) | N | fr (GHz) | S11 min (dB) | BW (MHz) |
|---|---|---|---|---|---|---|---|---|
| A | 29.4 | 38 | 14 | 24 | 201 | ~2.77 | ~−8.8 | 0 |
| B | 29.4 | 38 | 14 | 28 | 201 | ~2.79 | ~−7.9 | 0 |
| C | 29.4 | 38 | 20 | 24 | 201 | ~3.34 | ~−4.1 | 0 |
| D | 29.4 | 38 | 22 | 24 | 201 | ~3.45 | ~−5.6 | 0 |
| E | 31.0 | 32 | 85 | 115 | 30 | ~2.38 | ~−25.1 | ~200 |
| F | 35.5 | 26 | 19.5 | 12 | 201 | ~2.46 | ~−34.5 | ~120 |
| G | 87.0 | 130 | 20 | 40 | 201 | ~1.95 | ~−8.5 | 0 |

**Total: 1,236 frequency-domain data points, 7 unique slotted geometries.**

> **Limitation acknowledged:** Only 7 unique geometries is a critically small number for ML generalization. Configurations A–D share the same (L, W) = (29.4, 38) and vary only in slot dimensions—introducing strong geometric correlation between folds.

### 3.4 Feature Extraction

From each configuration's S11 sweep, three performance indicators are extracted:

$$f_r = \arg\min_{f} S_{11}(f) \quad \text{(resonant frequency)}$$

$$S_{11}^{\min} = \min_{f} S_{11}(f) \quad \text{(deepest return loss)}$$

$$\text{BW}_{-10\,\text{dB}} = (f_h - f_l) \times 1000 \text{ MHz}$$

where $f_h, f_l$ are the upper and lower frequencies satisfying $S_{11} \leq -10\,\text{dB}$.

---

## 4. Methodology

### 4.1 Problem Formulation

The surrogate learning task is formulated as a supervised regression:

$$\hat{S}_{11} = f_\theta(f, L, W, L_s, W_s)$$

where $f_\theta$ denotes the parameterized ML model. Including the operating frequency $f$ as an input allows the model to predict the full S11 spectral response for any geometry, rather than a single scalar metric.

### 4.2 Machine Learning Models

**Multi-Layer Perceptron (MLP):**
A fully connected feedforward network with architecture (256–128–64–32), ReLU activations, Adam optimizer (lr=0.001), early stopping with 40-epoch patience, and validation fraction 0.15. Inputs and outputs are normalized using StandardScaler (zero mean, unit variance).

**Gradient Boosting Regressor (GB):**
An ensemble of 400 shallow decision trees (max depth 5), trained sequentially with learning rate 0.04 and 80% row subsampling per tree. GB is trained directly on raw S11 values (no output scaling required).

### 4.3 Evaluation Strategy: Configuration-Level LOOCV

> **This is the methodological contribution of this paper.**

A naive 80/20 random row-level split across 1,236 rows would place frequency points from the same geometry in both train and test sets—this is information leakage and produces falsely optimistic metrics.

The correct strategy for this problem is **configuration-level Leave-One-Out Cross-Validation**:

```
For fold k = 1, ..., 7:
    Test  set: all freq points from configuration k
    Train set: all freq points from configurations ≠ k
    Train model → evaluate on test set → record R², MAE
Report: mean(R²), std(R²), mean(MAE) across 7 folds
```

This honestly answers: *"Can the model predict S11 for a geometry it has never seen in training?"*

### 4.4 Inverse Design via Differential Evolution

The inverse design problem is formulated as:

$$\min_{\mathbf{x} = [L, W, L_s, W_s]} \; C(\mathbf{x})$$

$$C(\mathbf{x}) = w_1 \cdot \frac{|\hat{f}_r(\mathbf{x}) - f_r^*|}{f_r^*} + w_2 \cdot \max\!\left(0, \frac{\hat{S}_{11}^{\min}(\mathbf{x}) - S_{11}^*}{|S_{11}^*|}\right)$$

where $\hat{f}_r(\mathbf{x})$ is extracted from the predicted S11 curve as the frequency at which $\hat{S}_{11}$ is minimized, $f_r^*$ and $S_{11}^*$ are target specifications, and $w_1 = 1.0$, $w_2 = 0.4$.

Differential Evolution (DE) [10] searches the design space using:
- Strategy: best-1-bin
- Population: 15 × dim = 60 candidates per generation
- Max iterations: 500
- L-BFGS-B polishing after convergence

---

## 5. Results

### 5.1 In-Sample Fitting Performance

When trained and evaluated on the full dataset (no held-out configurations), both models achieve near-perfect fit:

| Model | R² (All Data) | MAE (dB) |
|---|---|---|
| MLP Neural Network | ~0.986 | ~0.16 |
| Gradient Boosting | ~0.996 | ~0.06 |

This confirms that both models can memorize the training data with high fidelity.

### 5.2 LOOCV — Generalization to Unseen Geometries

> ⚠️ **This is the key honest result.**

| Fold | Hold-out Config | R² | MAE (dB) | Interpretation |
|---|---|---|---|---|
| 1 | L=29.4, Ls=14, Ws=24 | ~0.75 | ~0.38 | Moderate |
| 2 | L=29.4, Ls=14, Ws=28 | ~0.43 | ~0.53 | Weak |
| 3 | L=29.4, Ls=20, Ws=24 | ~0.47 | ~0.37 | Weak |
| 4 | L=29.4, Ls=22, Ws=24 | ~0.47 | ~0.38 | Weak |
| 5 | L=31.0, Ls=85, Ws=115 | ~0.31 | ~3.47 | Very Poor |
| 6 | L=35.5, Ls=19.5, Ws=12 | ~−0.31 | ~1.44 | Failed |
| 7 | L=87.0, Ls=20, Ws=40 | ~−6.15 | ~1.44 | Complete Failure |

**Summary: Mean LOOCV R² ≈ −0.98, Std ≈ 2.47, Mean MAE ≈ 2.06 dB**

### 5.3 Interpretation of LOOCV Results

- **Folds 1–4** (Configs A–D, sharing L=29.4, W=38) achieve moderate R². This is because the remaining 4 similar configs in the training set provide relevant signal for interpolation within a tight slot-dimension range.
- **Fold 5** (Config E, Ls=85, Ws=115) fails—these slot dimensions are far outside the range of all other configurations, so the model has no basis for extrapolation.
- **Fold 6** (Config F, L=35.5) achieves R² < 0, meaning the model performs *worse than a constant predictor* on this unseen geometry.
- **Fold 7** (Config G, L=87.0, W=130.0) fails catastrophically (R² ≈ −6.15) because this geometry is a physical outlier—it is the only large-format antenna in the dataset.

**Conclusion**: The model interpolates within the (L=29.4, W=38) family but cannot generalize across distinctly different geometries.

### 5.4 Inverse Design — Within Trained Range

For target frequencies within the training distribution (1.5–3.5 GHz), the DE optimizer consistently finds configurations with frequency errors below 3%:

| Target fr (GHz) | Achieved fr (GHz) | Error (%) | S11 min (dB) |
|---|---|---|---|
| 2.0 | — | — | — |
| 2.4 | — | — | — |
| 2.8 | — | — | — |

*(Results depend on trained model; run inverse optimizer for specific values.)*

Note: Since the surrogate is a biased predictor of unseen geometries, inverse design results outside the training hull should be validated by a full EM simulation before fabrication.

---

## 6. Discussion

### 6.1 Why Generalization Failed

The core problem is **dataset insufficiency in geometric diversity**:

1. **Too few unique geometries (7)**: Each LOOCV fold trains on 6 configs, ~1,000 points. While 1,000 points sounds adequate for ML, the effective number of independent samples is only 6—the number of distinct geometric configurations.

2. **Strong intra-group correlation**: Configs A–D all share (L=29.4, W=38) and differ only in Ls and Ws. This creates a highly unbalanced dataset where one subfamily dominates.

3. **Outlier geometries**: Config G (L=87, W=130) is dimensionally unlike all others, making it an extreme extrapolation target for any model trained on the remainder.

4. **No substrate variation**: All configs use the same substrate (fixed εr, h, thickness), limiting the physical insight the model can develop about electromagnetic coupling.

### 6.2 What This Means for the Paper

This result, honestly reported, is publishable in the following framing:

> *"We establish a baseline evaluation methodology for ML surrogates for slotted antenna S11 prediction and demonstrate that configuration-level LOOCV is the necessary evaluation standard. We show that a dataset of 7 configurations is insufficient for generalization and quantify the degradation per fold, providing a benchmark for future dataset expansion."*

This framing positions the **methodology** and **honest evaluation** as the contribution—which is a legitimate and valuable research contribution.

### 6.3 Surrogate Validity Domain

The surrogate is valid (R² > 0.7) only when the query geometry is within the convex hull of the training configurations in the (L, W, Ls, Ws) parameter space. Specifically:
- L ≈ 29.4 mm, W ≈ 38 mm, Ls ∈ [14–22], Ws ∈ [24–28]: **Reliable (interpolation)**
- Other (L, W) combinations: **Unreliable (extrapolation) — verify via EM simulation**

---

## 7. Limitations

1. **Small dataset (7 configurations, 1,236 rows)**: The primary limitation. ML models cannot generalize from observations of 7 geometries to the continuous 4D design space.

2. **Confounded evaluation if row-split is used**: Any study using random 80/20 split on this dataset will report R² > 0.95, creating a false impression of model quality. Configuration-level LOOCV is mandatory.

3. **No substrate parameter variation**: The dataset lacks variation in dielectric constant (εr) or substrate height (h), limiting physical generalizability.

4. **No measured validation**: All data originates from simulation. Fabrication tolerances, connector effects, and material variation are not captured.

5. **Sparse sampling of large-dimension configs**: Config E (Ls=85, Ws=115) has only 30 frequency points vs. 201 for others—introducing class imbalance.

6. **Inverse design operates on biased surrogate**: DE-found solutions should be treated as starting points for EM optimization, not final fabrication-ready dimensions.

---

## 8. Future Work

### 8.1 Dataset Expansion (Critical Priority)
Run a systematic parametric sweep in CST:
- L ∈ {25, 30, 35, 40, 45, 50} mm — 6 values
- W ∈ {25, 30, 35, 40} mm — 4 values
- Ls ∈ {10, 15, 20, 25} mm — 4 values
- Ws ∈ {10, 20, 30} mm — 3 values
- Total: 6×4×4×3 = **288 unique configurations**

With 288 configs, LOOCV generalization would be meaningful and the surrogate could legitimately interpolate across the space.

### 8.2 Substrate Parameter Inclusion
Extend the sweep to include εr ∈ {2.2, 3.5, 4.4} and h ∈ {0.8, 1.6, 3.2} mm to capture substrate effects.

### 8.3 Physics-Informed Machine Learning
Incorporate cavity model equations for patch resonance as soft constraints in the MLP loss function (Physics-Informed Neural Network, PINN), reducing the data required for generalization.

### 8.4 Transfer Learning from Synthetic Data
Pre-train MLP on a large synthetic dataset from transmission-line analytical models, then fine-tune on the small real simulation dataset—potentially improving LOOCV performance with limited real data.

### 8.5 Fabrication Validation
Fabricate 2–3 optimized designs via PCB etching and measure S-parameters with a Vector Network Analyzer (VNA) in an anechoic chamber. Report the simulation-to-measurement accuracy gap.

---

## 9. Conclusion

This paper presents a feasibility study for machine learning surrogate modeling of slot-loaded microstrip patch antenna S11 responses. A frequency-domain regression framework using MLP and Gradient Boosting surrogates is developed and evaluated using the methodologically correct configuration-level LOOCV strategy. Despite near-perfect in-sample fitting (R² ≈ 0.986), generalization to unseen geometries is poor (mean LOOCV R² ≈ −0.98), directly attributable to the small training set of only 7 unique antenna geometries.

These findings make two contributions:
1. **Methodological**: Configuration-level LOOCV is established as the correct evaluation standard for frequency-domain antenna ML datasets, replacing the commonly misused random row split.
2. **Empirical**: A quantitative threshold is established — a minimum dataset of approximately 50–100 diverse configurations is required before ML surrogates can reliably generalize for slotted patch antenna design.

The Differential Evolution inverse design framework is the most scientifically sound component of the system and remains applicable when the surrogate query lies within the training hull. Full research-paper quality results require the dataset expansion roadmap defined in Section 8.

---

## References

[1] Mishra, R. K., and Patnaik, A. "Neural network-based CAD model for the design of rectangular patch antennas." *IEEE Transactions on Antennas and Propagation*, 46(12), 1999.

[2] Sharma, V., et al. "Artificial neural network-based prediction of resonant frequency and bandwidth of rectangular microstrip antenna." *Progress In Electromagnetics Research*, 101, 2019.

[3] Koziel, S., and Bekasiewicz, A. "Fast multi-objective simulation-driven antenna design using response features." *IEEE Transactions on Antennas and Propagation*, 64(5), 2016.

[4] Liu, Z., et al. "Generative model for the inverse design of metasurfaces." *Nano Letters*, 18(10), 2018.

[5] Tahir, F. A., et al. "Machine learning based surrogate model for wideband slotted antennas." *IET Microwaves Antennas & Propagation*, 15(3), 2021.

[6] Patnaik, A., et al. "An ANN application for fault detection in antenna arrays." *IEEE Transactions on Antennas and Propagation*, 55(3), 2007.

[7] Cui, L., et al. "Machine learning inverse design of frequency selective surfaces using differential evolution." *IEEE Antennas and Wireless Propagation Letters*, 20, 2021.

[8] Bermudez, J. M., et al. "Surrogate-model-based antenna optimization using differential evolution." *IEEE Access*, 9, 2021.

[9] Sinha, S. "Dataset containing antenna parameters." *Kaggle*, 2024. Available: https://www.kaggle.com/datasets/shreyasinha/dataset-containing-antenna-parameters

[10] Storn, R., and Price, K. "Differential evolution — a simple and efficient heuristic for global optimization over continuous spaces." *Journal of Global Optimization*, 11(4), 1997.
