# Decisions log
Record every decision that changes a threshold, a contract field, a dataset choice or a locked decision.

| Date | Decision | Reason | Decided by |
| --- | --- | --- | --- |
| | Role names assigned | | all |
| | API contract frozen (version/date) | | Roles 2, 3, 4 |
| | Dataset path (A, B or C) | | all |
| | Extras chosen (in order) | | all |
| | Frontend: React + Vite or plain JavaScript | | Role 4 |
| | Hosts for backend and frontend | | Roles 3, 4 |
| 2026-11-07 | Noise model: PROPOSED (N/P/K ±10% with floor 2.0 kg/ha, temp ±1.5°C, humidity ±5 pts, pH ±0.3, rainfall ±10% with floor 5.0 mm) | Baseline physical measurement uncertainty for stability (E8) and calibration robustness (E4). Formal review pending with Role 1. | Role 2 (proposed to Role 1) |
| | Noise model (final values) | | Roles 1, 2 |

---

## Detailed Specifications

### Proposed Physical Noise Model (Role 2 -> Role 1 Review)

*Status: PROPOSED / PENDING CONFIRMATION WITH ROLE 1*
*Draft Artifact: `backend/models/noise_model.json`*

| Feature | Scale Type | Proposed Tolerance | Additive Floor | Distribution | Physical Clipping Bounds | Engineering & Agronomic Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Nitrogen (N)** | Relative | $\pm 10\%$ | $2.0\text{ kg/ha}$ | Gaussian ($\sigma = 0.05 \cdot N$) | $[0.0, 200.0]\text{ kg/ha}$ | Lab extraction (Kjeldahl/optical) exhibits 5–15% uncertainty. Relative scaling fits concentration variance; floor prevents zero variance at zero. |
| **Phosphorus (P)** | Relative | $\pm 10\%$ | $2.0\text{ kg/ha}$ | Gaussian ($\sigma = 0.05 \cdot P$) | $[0.0, 200.0]\text{ kg/ha}$ | Bray/Olsen extraction method variance across testing labs is typically $\pm 10\%$. Floor handles depleted soil inputs. |
| **Potassium (K)** | Relative | $\pm 10\%$ | $2.0\text{ kg/ha}$ | Gaussian ($\sigma = 0.05 \cdot K$) | $[0.0, 300.0]\text{ kg/ha}$ | Flame photometry and ammonium acetate extraction variance. Upper bound extended to 300 to accommodate high-K crops (e.g. grapes, apples). |
| **Temperature** | Absolute | $\pm 1.5\text{ }^\circ\text{C}$ | N/A | Gaussian ($\sigma = 0.75\text{ }^\circ\text{C}$) | $[-10.0, 60.0]\text{ }^\circ\text{C}$ | Canopy microclimate vs nearest meteorological station discrepancy. Interval temperature scale requires absolute rather than percentage noise. |
| **Humidity** | Absolute | $\pm 5.0\text{ pts}$ | N/A | Gaussian ($\sigma = 2.5\text{ pts}$) | $[0.0, 100.0]\%$ | Farm capacitive sensor calibration drift is typically $\pm 3$ to $\pm 5$ percentage points across relative humidity ranges. |
| **pH** | Absolute | $\pm 0.3$ | N/A | Gaussian ($\sigma = 0.15$) | $[0.0, 14.0]$ | Routine glass electrode slurry testing variation. Logarithmic scale ($\text{pH} = -\log_{10}[\text{H}^+]$) makes relative % meaningless; $0.3$ is a $\sim 2\times$ ion shift. |
| **Rainfall** | Relative | $\pm 10\%$ | $5.0\text{ mm}$ | Gaussian ($\sigma = 0.05 \cdot \text{rain}$) | $[0.0, 1000.0]\text{ mm}$ | Localized convective precipitation variation across a farm plot. Additive floor of $5.0\text{ mm}$ accounts for light shower recording thresholds. |

#### Implementation & Reproducibility Rules:
1. **Gaussian Sampling:** Sample from zero-mean Gaussian $\mathcal{N}(0, \sigma^2)$ where $\sigma = \frac{\text{Tolerance}}{2}$, guaranteeing that $\sim 95.45\%$ of generated perturbations fall within the agreed $\pm$ tolerance.
2. **Relative Scaling with Additive Floor:** For relative features $X \in \{N, P, K, \text{rainfall}\}$, the standard deviation is:
   $$\sigma_X = \max\left(\frac{\text{tolerance}}{2} \times X, \frac{\text{floor}}{2}\right)$$
3. **Strict Physical Boundary Clipping:** All perturbed values are clipped via `np.clip(noisy_X, LOW, HIGH)` matching the API physical limits.
4. **Reproducibility:** Seeded sampling using `numpy.random.default_rng(seed=42)` across 200 batch copies in a single vectorized array operation (`PROJECT_GUIDE.md` Section 7.3 Stage 4).

| | Risk profile thresholds (final values) | | Role 2 |
