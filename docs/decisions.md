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
