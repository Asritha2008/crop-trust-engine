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
