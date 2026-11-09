# Decisions log
Record every decision that changes a threshold, a contract field, a dataset choice or a locked decision.

| Date | Decision | Reason | Decided by |
| --- | --- | --- | --- |
| | Role names assigned | | all |
| | API contract frozen (version/date) | | Roles 2, 3, 4 |
| | Dataset path (A, B or C) | | all |
| | Extras chosen (in order) | | all |
| 2026-11-07 | Frontend: React + Vite selected | Clean component separation across the 5 pages, seamless chart library integration, and fast phone-friendly builds | Role 4 |
| 2026-11-08 | Hosts: backend on Render (free, Docker); frontend on Vercel | Free tiers. The backend sleeps when idle, so the frontend shows a waking-server note | Roles 3, 4 |
| 2026-11-07 | Noise model (final values): APPROVED (N/P/K 10% with floor 2 units, temp 1.5°C, humidity 5 pts, pH 0.3, rainfall 10% with floor 5 units; 1-sigma interpretation) | Approved by Role 1 (Abhishek) as primary 1-sigma standard deviation for E4 calibration stress testing and E8 stability. | Role 1 (Abhishek) & Role 2 |
| 2026-11-08 | Dataset units: N, P, K are unitless "ratio of content in soil"; rainfall is in mm with the time period NOT stated; temperature °C, humidity %, pH as listed. Data "built by augmenting datasets of rainfall, climate and fertilizer data available for India". Licence: Apache 2.0. | Source: Kaggle Data Card (atharvaingle). Do not use ppm or kg/ha anywhere. | Role 1 |
| 2026-11-08 | Final model = lowest post-calibration log-loss on VAL; ties broken by ECE, then noisy accuracy. | Prioritises honest confidence and calibration (Handbook §5.4) for downstream trust engine stages. | Role 1 |
| 2026-11-08 | Amendment: final model = Extra Trees (isotonic). Original rule (lowest clean val log-loss) would have picked Gaussian NB isotonic (0.0116 vs 0.0147). | Rule changed after seeing results. Over 5 noise seeds (42-46, noise.py 1 sigma, same 330 val rows) ET isotonic vs GNB isotonic: noisy log-loss lower in 5/5 seeds (mean diff -0.49), noisy ECE lower in 5/5, noisy accuracy higher in 4/5. ET isotonic has the lowest mean noisy ECE of the nine variants (0.027) but not the lowest noisy log-loss (ET sigmoid 0.246 vs ET isotonic 0.424 +/- 0.159); ET sigmoid is far worse calibrated on clean data (ECE 0.082 vs 0.008). Caveats: seeds are noise draws on the same val rows, not independent data. | Role 1 |
| 2026-11-08 | Amendment: final model = Extra Trees (isotonic). Original rule (lowest clean val log-loss) would have picked Gaussian NB isotonic (0.0116 vs 0.0147). | SUPERSEDED by the 5-seed amendment row above (the 'best noisy log-loss of all nine' claim held only at seed 42; over seeds 42-46 ET isotonic noisy log-loss is 0.424 +/- 0.159, not the lowest). Original text kept for the audit trail: Rule changed after seeing results. Under simulated noise (noise.py, 1 sigma, seed 42, one draw, 330 val rows) ET isotonic had the best noisy log-loss of all nine variants (0.2368; GNB isotonic 0.7476). Noisy ECE 0.0263 (RF isotonic 0.0214 was lower). Caveats: single noise draw; 1-2 validation errors per model; clean differences untested. | Role 1 |
| 2026-11-08 | In-browser familiarity check is a distance-only estimate; the server stays the final authority | The server verdict also uses stability and other stages. ood_browser.json is a copy from Varshith's branch (commit 9e0c912) | Role 4 |
| 2026-11-08 | Climate stress-test scenario shifts are placeholders until Role 1 confirms them | Not yet taken from Role 1's data | Role 4 |
| 2026-11-09 | Units wording: N, P, K = dataset units (unitless ratios); rainfall = mm (period not stated); data described as augmented, not synthetic. | Source: Kaggle Data Card (atharvaingle) | all |

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


| | Risk profile thresholds (final values) | | Role 2 |


## 2026-11-08: Backend hosting, units, batch limit (Role 4)
- Backend on Render (free, Docker): https://crop-trust-engine.onrender.com. First request after idle can take up to a minute, so the frontend shows a "waking server" note.
- Frontend origin for CORS: https://crop-trust-engine-frontend.vercel.app (set via FRONTEND_URL).
- Units: N, P, K unitless (hidden in UI); rainfall "mm" with no period; temperature °C, humidity %, pH shown.
- Batch upload limit is 50 rows (error "At most 50 rows per upload"); the Batch page will read the limit from config.js.