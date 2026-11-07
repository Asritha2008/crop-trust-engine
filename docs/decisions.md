# Decisions log
Record every decision that changes a threshold, a contract field, a dataset choice or a locked decision.

| Date | Decision | Reason | Decided by |
| --- | --- | --- | --- |
| | Role names assigned | | all |
| | API contract frozen (version/date) | | Roles 2, 3, 4 |
| | Dataset path (A, B or C) | | all |
| | Extras chosen (in order) | | all |
| 2026-11-07 | Frontend: React + Vite selected | Clean component separation across the 5 pages, seamless chart library integration, and fast phone-friendly builds | Role 4 |
| | Hosts for backend and frontend | | Roles 3, 4 |
| 2026-11-07 | Noise model (final values): APPROVED (N/P/K 10% with floor 2 units, temp 1.5°C, humidity 5 pts, pH 0.3, rainfall 10% with floor 5 units; 1-sigma interpretation) | Approved by Role 1 (Abhishek) as primary 1-sigma standard deviation for E4 calibration stress testing and E8 stability. | Role 1 (Abhishek) & Role 2 |
| 2026-11-07 | OOD detector selection: Per-crop Mahalanobis with Ledoit-Wolf shrinkage chosen over Isolation Forest and Range Check | Outperformed all baselines on manufactured unknowns (Box AUROC 0.9996 vs. 0.5000 range check; Shuffled AUROC 0.9778 vs. 0.5987 iforest; Corrected Mean LOCO AUROC 0.9797). Scaled with train-fitted StandardScaler. | Role 2 |
| 2026-11-07 | OOD threshold calibration: Empirical out-of-fold (OOF) percentiles from 1,540 training rows (Cons: 11.71 / 97%, Bal: 12.92 / 99%, Bold: 13.75 / 99.5%, Caution: 10.78 / 95%) | Provides a stable 1,540-point empirical reference distribution preventing small-sample validation noise (Handbook 7.3). Calibrated using strict train-OOF separation from independent evaluation sets. | Role 2 |
| 2026-11-07 | Stage 2 Pure Interface & Interim API Handoff: `assess_ood()` + `StagesNotImplementedError` (HTTP 501) | Approved with Role 3. Pure Stage 2 OOD diagnostics exposed via `assess_ood()`; `assess()` returns contract refusal on OOD rejection and raises `StagesNotImplementedError` on acceptance (mapped by Role 3 to HTTP 501). Zero fabrication of uncalibrated probabilities, stability, or trust scores. | Role 2 & Role 3 |
| 2026-11-08 | Dataset units: N, P, K are unitless "ratio of content in soil"; rainfall is in mm with the time period NOT stated; temperature °C, humidity %, pH as listed. Data "built by augmenting datasets of rainfall, climate and fertilizer data available for India". Licence: Apache 2.0. | Source: Kaggle Data Card (atharvaingle). Do not use ppm or kg/ha anywhere. | Role 1 |
| 2026-11-08 | Final model = lowest post-calibration log-loss on VAL; ties broken by ECE, then noisy accuracy. | Prioritises honest confidence and calibration (Handbook §5.4) for downstream trust engine stages. | Role 1 |
| 2026-11-08 | Amendment: final model = Extra Trees (isotonic). Original rule (lowest clean val log-loss) would have picked Gaussian NB isotonic (0.0116 vs 0.0147). | Rule changed after seeing results. Under simulated noise (noise.py, 1 sigma, seed 42, one draw, 330 val rows) ET isotonic had the best noisy log-loss of all nine variants (0.2368; GNB isotonic 0.7476). Noisy ECE 0.0263 (RF isotonic 0.0214 was lower). Caveats: single noise draw; 1-2 validation errors per model; clean differences untested. | Role 1 |
| 2026-11-08 | Stability evaluation (E8) finalized on calibrated Extra Trees model | Evaluated 330 validation inputs under M=200 physical perturbations (66,000 points). Mean stability 94.45%, flip rate 5.55%, overall noisy accuracy 94.53%. Inputs with stability <80% drop to 66.95% accuracy (11.7x relative error spike). Documented 80% as empirical evaluation cutoff. | Role 2 |

---

## Detailed Specifications

### Physical Measurement-Noise Model

*Status: FINAL / APPROVED by Role 1 (Abhishek) & Role 2*  
*Artifact: `backend/models/noise_model.json`*  
*Implementation: `backend/noise.py` (`apply_noise`, `compute_feature_sigmas`)*

| Feature | Scale Type | Primary $1\sigma$ Formulation | Additive $1\sigma$ Floor | Physical API Bounds | Rationale & Behavioral Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Nitrogen (N)** | Relative | $\sigma = \max(0.10 \cdot \|N\|, 2.0)$ | $2.0\text{ dataset units}$ | $[0.0, 200.0]$ | Relative scaling matches concentration variance. Additive floor of 2 dataset units prevents zero-variance collapse on depleted soil values. |
| **Phosphorus (P)** | Relative | $\sigma = \max(0.10 \cdot \|P\|, 2.0)$ | $2.0\text{ dataset units}$ | $[0.0, 200.0]$ | Relative scaling with 2 dataset unit floor prevents zero-variance at boundaries. |
| **Potassium (K)** | Relative | $\sigma = \max(0.10 \cdot \|K\|, 2.0)$ | $2.0\text{ dataset units}$ | $[0.0, 300.0]$ | Relative scaling with 2 dataset unit floor. Upper bound accommodates high-K crops. |
| **Temperature** | Absolute | $\sigma = 1.5$ | N/A | $[-10.0, 60.0]\text{ }^\circ\text{C}$ | Fixed additive standard deviation representing microclimate and sensor variability. Interval scale requires absolute noise. |
| **Humidity** | Absolute | $\sigma = 5.0$ | N/A | $[0.0, 100.0]\text{ pts}$ | Fixed additive standard deviation in percentage points representing capacitive sensor drift. |
| **pH** | Absolute | $\sigma = 0.3$ | N/A | $[0.0, 14.0]$ | Fixed additive standard deviation. Logarithmic scale ($-\log_{10}[\text{H}^+]$) makes absolute noise mathematically and physically correct. |
| **Rainfall** | Relative | $\sigma = \max(0.10 \cdot \|\text{rainfall}\|, 5.0)$ | $5.0\text{ dataset units}$ | $[0.0, 1000.0]$ | Relative precipitation scaling across farm topography. Floor of 5 dataset units prevents zero-variance during dry periods. |

#### Implementation & Evaluation Rules:
1. **1-Sigma Interpretation:** Magnitudes represent standard deviations ($\sigma$), directly producing independent zero-mean Gaussian perturbations:
   $$x_{i,\text{noisy}} = x_i + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma_i^2)$$
2. **Independent Feature Assumption:** Perturbations across the 7 features are sampled independently.
3. **Noisy Copies are Evaluation-Only:** Noisy copies are strictly used for evaluation (E4 noisy calibration robustness, E8 stability flip rates, and climate sensitivity curves). Noisy data is NEVER used to train classifiers, fit calibrators, or compute OOD covariance matrices.
4. **Physical Boundary Clipping:** Perturbed values are clipped to API physical bounds via `np.clip(noisy, LOW, HIGH)`. Boundary clipping counts are exposed via `return_clip_stats=True` in `backend/noise.py` to monitor edge effects.
5. **Reproducibility & Vectorization:** Seeded via `numpy.random.default_rng(seed=42)` across $N$ rows and $M$ samples in a single vectorized NumPy batch operation without modifying original inputs in-place.


### Out-of-Distribution (OOD) Detector Architecture & Calibrated Thresholds

*Status: COMPLETED (Evaluation & Artifacts) / Integration Pending*
*Evaluation Notebook: `notebooks/04_ood_eval.ipynb`*
*Artifacts: `backend/models/ood_stats.joblib`, `backend/models/iforest.joblib`, `backend/models/thresholds.json`*

#### 1. Detector Selection Rationale
In agricultural crop recommendation, feature distributions are multi-modal, forming 22 distinct crop clusters in $\mathbb{R}^7$. Four detection architectures were implemented and systematically compared:
- **Per-Crop Mahalanobis Distance with Ledoit-Wolf Shrinkage:** Measures multivariate distance to the nearest crop cluster cloud. Outperformed all alternatives across both Box-Uniform unknowns (AUROC **0.9996**, FPR@95% **0.0000**) and Shuffled-Column unknowns (AUROC **0.9778**, FPR@95% **0.0727**).
- **Isolation Forest ($n=200$):** Struggled on broken feature correlations (Shuffled AUROC **0.5987**, FPR@95% **0.9000**), because axis-aligned random tree cuts fail to capture oblique covariance boundaries. Retained as secondary serialized benchmark artifact (`backend/models/iforest.joblib`).
- **kNN Distance ($k=5$):** Achieved strong box detection (AUROC **0.9989**) and moderate shuffled detection (AUROC **0.9479**), but exhibits higher computational inference cost than covariance projection.
- **1D Range Check (Naive Baseline):** Completely failed on joint combinations (AUROC **0.5000**, coin-flip on both Box and Shuffled unknowns), confirming the foundational premise of Section 5.3 that individual range checks cannot detect impossible multivariate combinations.

#### 2. Scaling & Covariance Regularization
- **Feature Standardization:** Features differ by orders of magnitude (rainfall variance $\gg$ pH variance). A `StandardScaler` fitted strictly on $X_{\text{train}}$ scales inputs to isotropic variance prior to Ledoit-Wolf shrinkage, ensuring the shrinkage target ($\frac{\text{tr}(\Sigma)}{p} I$) treats all features equitably.
- **Ledoit-Wolf Shrinkage:** With $N_c = 70$ training rows per crop and $p = 7$ features, empirical sample covariances risk ill-conditioning. Analytical shrinkage estimation guarantees positive definiteness and invertible precision matrices across all 22 classes without arbitrary manual jitter. Mean shrinkage parameter $\bar{\lambda} = 0.0758$ (min $0.0193$, max $0.1807$).

#### 3. Calibration vs. Evaluation Protocol
- **Zero Leakage:** The test set (`test_idx.npy`, 330 rows) remains 100% untouched.
- **Out-of-Fold (OOF) Reference Distribution:** 5-fold stratified cross-validation on $X_{\text{train}}$ produces 1,540 genuine held-out Mahalanobis distances. This replaces noisy small-sample percentiles (where the 99th percentile of 330 validation rows rests on only 3 samples) with an empirical reference curve resting on 15+ tail samples.
- **Independent Synthetic Seeds:** Calibration unknowns (Box `seed=101`, Shuffled `seed=303`) were strictly quarantined from independent evaluation unknowns (Box `seed=202`, Shuffled `seed=404`) and held-out validation rows (`val_eval`, 110 rows).

#### 4. Calibrated Threshold Profiles (`backend/models/thresholds.json`)

| Risk Profile | Refuse Cutoff ($d^2$) | Refuse Percentile | Caution Cutoff ($d^2$) | Caution Percentile | Val False Alarm (Eval) | Box Unknown Det. | Shuffled Unknown Det. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Conservative** | $11.71$ | $97.0\%$ | $9.53$ | $90.0\%$ | $5.5\%$ | $99.8\%$ | $94.4\%$ |
| **Balanced (Default)** | $12.92$ | $99.0\%$ | $10.78$ | $95.0\%$ | $2.7\%$ | $99.8\%$ | $92.7\%$ |
| **Bold** | $13.75$ | $99.5\%$ | $11.71$ | $97.0\%$ | $2.7\%$ | $99.8\%$ | $91.2\%$ |

#### 5. Leave-One-Crop-Out (LOCO) Robustness
Across 22 distinct LOCO evaluations (training scaler and Ledoit-Wolf detectors strictly on 21 crops [1,470 rows], evaluating on 15 validation rows of the held-out crop vs. 315 validation rows of known crops; test set 100% excluded):
- **Corrected Mean LOCO AUROC:** $0.9797 \pm 0.0526$ (Mean FPR@95% TPR: $0.1046 \pm 0.2338$).
- **Comparison with Initial Exploration:** Earlier exploration used all 100 rows ($0.9807 \pm 0.0453$), which inadvertently included 15 test-set rows of the held-out crop. Correcting this to strictly validation-only rows confirms an almost identical AUROC ($\Delta = 0.0010$) while strictly preserving test-set hygiene.
- **Easiest Crops (AUROC 1.0000):** `apple`, `banana`, `chickpea`, `coconut`, `coffee`, `watermelon`, `grapes` (highly isolated agronomic requirements).
- **Hardest Crops:** `jute` (AUROC $0.7934$), `lentil` (AUROC $0.8521$), `kidneybeans` (AUROC $0.9596$), `rice` (AUROC $0.9733$). `jute` and `rice` overlap heavily in high-rainfall / high-humidity regimes, accurately reflecting agronomic reality rather than detector failure.

#### 6. Midpoint-Trap Findings (Handbook 7.4)
- **Close Clusters (`rice` / `jute`):** The midpoint lands within known high-rainfall density ($d^2 = 1.07$, 0.0th percentile relative to OOF), correctly classified as familiar.
- **Distant Clusters (`lentil` / `chickpea`, `apple` / `grapes`):** The midpoint lands in sparse intermediate space ($d^2 = 22.67$, 100.0th percentile), triggering OOD refusal as anticipated.

#### 7. Known Limitations
- Synthetic unknowns (box-uniform and column-shuffled) serve as proxies for true field failure modes (sensor drifts, out-of-season planting, unmeasured soil conditions). Real field anomalies may exhibit localized structure.
- Overlapping crops (e.g., rice and jute) cannot be completely separated by OOD density alone; ambiguity between them is safely handled downstream by Stage 3 confidence margins and Stage 5 caution verdicts.
- **Provisional Nature of Thresholds:** All numerical thresholds (11.71, 12.92, 13.75, 10.78, 9.53) are provisional cutoffs tuned on training OOF empirical distributions and synthetic unknowns. They are subject to refinement when real field sensor validation occurs.
- **Agronomic vs. Statistical Suitability:** Passing the OOD gate does NOT establish agronomic suitability or crop success. It merely confirms that the input's environmental and soil features fall within familiar statistical density clouds observed in the training data.
- **Uncalibrated Placeholder Model:** The current baseline Random Forest (`backend/models/crop_pipeline.joblib`) is uncalibrated. Its raw probabilities are NOT used for OOD detection, trust scoring, or selective verdicts.

#### 8. Pure Stage 2 Interface & Interim API Handoff Architecture
*Status: APPROVED by Role 2 & Role 3 (Ashritha)*

To preserve strict architectural isolation between the 5 stages and prevent the emission of uncalibrated probabilities or hardcoded placeholder scores (such as 0.94 stability or 88 trust score), a dual-interface architecture is implemented:
1. **Pure Stage 2 Method (`TrustEngine.assess_ood(row, risk)`):**
   - Implements Stage 1 schema validation, numeric finite checks, and physical bounds.
   - Computes empirical per-crop Ledoit-Wolf Mahalanobis distances, OOF percentile rank, and risk thresholding.
   - Returns a pure diagnostic dictionary: `eligible` (bool), `flagged` (bool), `percentile` (float), `min_distance` (float), `threshold` (float), `caution_threshold` (float), `nearest_known` ([c1, c2]), `unusual_pair` ([feat_a, feat_b] | None), and `risk_profile` (str).
   - Contains zero downstream predictions, zero probabilities, zero stability scores, and zero trust scores.
2. **Contract Endpoint Method (`TrustEngine.assess(row, risk)`):**
   - Calls `assess_ood(row, risk)`.
   - **OOD Rejection (`flagged=True`):** Returns the complete frozen API contract response for `verdict: "refuse"` (HTTP 200) with required explicit `null` optional fields, `top_crops: []`, and `trust_score: 0`.
   - **OOD Acceptance (`eligible=True`):** Raises `StagesNotImplementedError("Stages 3-5 not integrated yet", ood_result=...)`.
   - **Role 3 FastAPI Mapping:** Role 3 catches `StagesNotImplementedError` and returns `HTTP 501 Not Implemented` with payload `{"error": "Stages 3-5 not integrated yet"}`. This adheres to Section 1.6 of `docs/api-contract.md` without fabricating uncalibrated recommendations.

#### 9. Bivariate Excess-Coupling Diagnostic Heuristic (`_find_unusual_pair`)
*Status: IMPLEMENTED & VERIFIED*

- **Mathematical Nature:** The 2D sub-covariance Mahalanobis distance is a 2D marginal subspace projection heuristic, **not** an exact orthogonal decomposition of the full 7D Mahalanobis distance. In linear algebra, $(\Sigma_{(i,j)})^{-1}$ is the inverse of a principal submatrix, not a submatrix of the full inverse precision matrix $\Sigma^{-1}$.
- **Excess Bivariate Coupling Metric:** Candidate pairs are ranked by divergence beyond what independent marginal deviations would predict:
  $$\Delta d_{ij}^2 = d_{ij}^2 - (z_i^2 + z_j^2)$$
  where $z_i, z_j$ are standardized marginal distances relative to the nearest crop centroid and standard deviations.
- **Selection Criteria:**
  1. Excess coupling: $\Delta d_{ij}^2 \ge 0.5$ (demonstrates true correlation breakdown).
  2. Statistical significance: $d_{ij}^2 \ge 5.99$ ($\chi_2^2$ 95th percentile).
  3. Bounded marginals: $|z_i| \le 3.5$ and $|z_j| \le 3.5$ (prevents single univariate outliers from polluting all associated pairs).
- **Honest Null Fallback:** When an anomaly is diffuse across higher dimensions (such as the lentil-chickpea midpoint void) or lacks statistically significant excess coupling, `_find_unusual_pair()` returns `None` (`null` in JSON). It never invents a spurious pair attribution.
- **Non-Causal Explanation:** Explanations describe observed bivariate divergence (e.g. *"Features 'humidity' (86.0) and 'rainfall' (125.0) exhibit high bivariate divergence relative to rice patterns"*), explicitly avoiding causal claims that the pair caused crop failure.

### Stability Evaluation (E8) on Calibrated Extra Trees Model

- **Status:** FINAL EVALUATION / VERIFIED on Calibrated Model
- **Model Artifact:** `backend/models/crop_pipeline_calibrated.joblib` (CalibratedClassifierCV wrapping ExtraTreesClassifier with isotonic calibration, cv=5)
- **Notebook:** `notebooks/05_stability.ipynb`
- **Summary:** `docs/figures/stability_provisional_summary.json`

#### Methodology
- **Evaluation Split:** 330 validation rows (`data/splits/val_idx.npy`, 15 samples per class across 22 crops). Test set (`test_idx.npy`, 330 rows) remains completely untouched (zero leakage).
- **Perturbation Protocol:** $M = 200$ noisy replicas per validation row ($66,000$ perturbed test instances) generated via `backend.noise.apply_noise` using the approved primary $1\sigma$ noise model with physical boundary clipping and fixed random seed 42.
- **Model Ingestion:** Inputs passed as unscaled raw features in canonical order `['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']`.

#### Key Empirical Results
- **Clean Validation Accuracy:** $99.39\%$ ($328/330$ correct; errors only on row 83 [rice misclassified as jute] and row 2049 [jute misclassified as rice]).
- **Mean Stability:** $94.45\%$ (median $99.00\%$, standard deviation $10.50\%$).
- **Mean Flip Rate:** $5.55\%$ (under $1\sigma$ physical noise).
- **Noisy Accuracy Across All $66,000$ Perturbations:** $94.53\%$.
- **Fully Stable Inputs ($100\%$ retention over 200 samples):** $131/330$ ($39.7\%$).
- **Fragile Inputs ($< 80\%$ stability):** $29/330$ ($8.79\%$).

#### Empirical Evaluation of 80% Stability Cutoff
Segmenting the validation set by the $80\%$ stability cutoff demonstrates a notable contrast in error susceptibility under noise:
- **Stable Inputs ($\ge 80\%$ stability, $n = 301$):** Noisy accuracy is **$97.18\%$** (error rate $2.82\%$).
- **Fragile Inputs ($< 80\%$ stability, $n = 29$):** Noisy accuracy drops to **$66.95\%$** (error rate $33.05\%$).
- **Relative Error Risk Spike:** $\frac{33.05\%}{2.82\%} = \mathbf{11.74\times}$ (exact raw ratio: $11.74\times$; ratio of rounded rates $33.05\% / 2.82\% \approx 11.72\times$).

**Scientific & Methodological Nuance:**
- The $80\%$ threshold serves strictly as an **empirical evaluation segmentation cutoff** for analyzing error concentration under physical perturbations.
- While the observed validation results support its usefulness by showing where errors concentrate, the low-stability subgroup contains only **29 samples** ($8.79\%$ of the validation split).
- These results are based on **simulated physical noise** and **a single validation split** ($N = 330$).
- This evaluation **does not constitute final approval of the TrustEngine production stability threshold**, which remains subject to Stage 4/5 integration, risk profile tuning, and final team sign-off.

#### Crop Fragility Profile
- **Most Fragile Crops:** `jute` (mean stability $73.87\%$, flip rate $26.13\%$), `lentil` ($81.50\%$), `rice` ($82.90\%$), `blackgram` ($86.27\%$). Note: `jute` and `rice` overlap closely in the high-humidity, high-rainfall feature region.
- **Most Stable Crops:** `chickpea` ($100.0\%$), `orange` ($100.0\%$), `coffee` ($99.70\%$), `mango` ($99.57\%$).

#### Figures & Artifacts Generated
- `docs/figures/stability_distribution.png`: Histogram and CDF of validation stability scores showing concentration near 1.0.
- `docs/figures/stability_per_crop.png`: Horizontal bar chart ranking all 22 crops by mean stability under physical noise.
- `docs/figures/sensitivity_curves.png`: Prediction retention curves sweeping individual features across $[-3\sigma, +3\sigma]$.
