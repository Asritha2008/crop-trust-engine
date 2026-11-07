# Role 2: Trust Engine Engineer: OOD, stability, verdict, explanations

Also read: AGENTS.md. Handbook sections: 2 (Role 2), 3.5, 5.3 to 5.8, 7.2 to 7.4, 7.6 (E5 to E9), 7.10, 8.3A.

## Mission
Build the Trust Engine so the system can refuse, hedge and explain, and prove it with measurements. This is the idea that sets the project apart.

## You own (edit freely)
notebooks/04_ood_eval, 05_stability; backend/trust_engine.py; backend/models/ ood_stats.joblib, iforest.joblib, thresholds.json, noise_model.json, conformal.json; the in-browser statistics file (ood_browser.json); explanation functions; backend/tests/test_engine.py.

## Tasks, in order (pick one at a time)
1. Week 1: with Role 3 agree the class interface: TrustEngine.load(dir) and assess(row, risk) returning exactly the contract JSON. Publish a stub trust_engine.py returning the three mock replies by a trivial rule so the API and website can be built immediately.
2. Week 1: agree the noise model with Role 1 (suggested pH +/-0.3, temperature +/-1.5 C, humidity +/-5 points, N/P/K/rainfall +/-10%); write the reasons down.
3. OOD detectors, fitted on training rows only: per-crop Mahalanobis with Ledoit-Wolf (minimum over crops), Isolation Forest, kNN-distance baseline, and the plain min/max range check as the baseline to beat. The fitting function must take a feature list so the weather-only gate can reuse it.
4. E5 manufactured unknowns: leave-one-crop-out (several crops), box-uniform random inputs, shuffled-column inputs. Report AUROC and FPR at 95% TPR for all four detectors, which held-out crops were missed and why, and show the range check accepting impossible combinations that the joint detector refuses.
5. E6 thresholds: use out-of-fold scores (the 99th percentile of 330 rows rests on about 3 rows). Plot false-alarm vs detection, mark the chosen point, write thresholds.json for the three risk profiles.
6. Midpoint trap: decide how a point halfway between two crops is treated (consider a kNN second gate) and write the decision down.
7. E8 stability: noise function, 200 noisy copies predicted in ONE array call, per-feature sensitivity curves, flip rate. Show that low-stability answers err more often (use noisy test inputs).
8. Verdict logic: decide() and trust_score() (handbook 7.3 stage 5), three risk profiles. E7 risk-coverage curve on clean, noisy and manufactured-unknown inputs.
9. Explanations for POST /explain: unusual_pair, counterfactual and smallest_flip (decision margin), measure_next, why-not-second-crop, comfort-zone percentiles, nearest real examples, ensemble disagreement. Word messages as relative to the nearest crop.
10. E9 conformal sets (bonus): empirical coverage and average set size.
11. Exports: ood_browser.json (class means and inverse covariances) for Role 4; psi() helper for Role 3; weather-only gate for Role 1.
12. Unit tests for the engine, runnable without a web server.

## You hand over
Stub engine (week 1), real engine and artefacts (end of phase 1), ood_browser.json, psi helper, weather-only gate, evidence figures (AUROC table, risk-coverage curve).

## You need
Splits and baseline pipeline from Role 1 (day 3 to 4), final pipeline later; the frozen contract from Role 3.

## Role-specific rules for the AI
- The OOD detector learns normal data only. Never train it on unusual examples; those are for testing it.
- Thresholds come from validation (ideally out-of-fold) scores, never the test set.
- Distinguish invalid input (422, Role 3's job) from unusual-but-valid input (200, verdict refuse, your job).
- Percentiles, 0.70, 0.20, 0.80 and noise sizes are placeholders until tuned.
- Report where the detector fails; a detector that never fails invites suspicion.
- Keep trust_engine.py free of web code.

## First prompt to paste
> I'm Role 2 (Trust Engine Engineer). Read AGENTS.md, docs/roles/ROLE2_Trust_Engine.md, and handbook sections 2 (Role 2), 5.3 to 5.8, 7.3 and 7.6. Don't write code yet. Summarise my tasks, the files I own, what I need from others, and what you'd build first. Then wait for me to pick one task.
