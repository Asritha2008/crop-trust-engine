# Crop Trust Engine: Team Plan and Project Handbook

Nov 7, 2026 · @varshith

## 1. The project and the four roles at a glance

We are building a crop recommender that knows when not to trust itself, explains why, and proves that behaviour with measurements. Four people share the work in four lanes that meet only at frozen interfaces, so everyone can work from a different place without waiting.

**Pitch for the mentor:** "Most groups will build a crop recommender. We build one that knows when not to trust itself, tells the user why, and proves that behaviour with measurements."

**Repository:** [github.com/Asritha2008/crop-trust-engine](https://github.com/Asritha2008/crop-trust-engine)

**Course context:** Web Development project; stack named by faculty is JavaScript and ML; ML model is the most important part; doing only the minimum will not pass. Deliverables: a trained and evaluated model, a deployed API, a user-facing website or app, a report, a demo. Official problem: build a crop classifier from soil nutrients and weather variables, compare classifiers, measure class-wise performance, test recommendation stability under input changes, and add a mechanism that flags unusual combinations instead of confidently recommending a crop for every input.

### Locked decisions (do not change without asking the whole team)

- Classifier trained on the Kaggle Crop Recommendation data (2,200 rows, 22 crops x 100 rows; features N, P, K, temperature, humidity, pH, rainfall); 9 models compared properly.
- Trust Engine with five stages: validate, OOD gate, calibrated classifier, stability, verdict.
- Three verdicts: Recommend, Recommend with caution, Refuse, each with a reason.
- Per-crop Mahalanobis plus Isolation Forest as OOD detectors, proven with manufactured unknowns.
- FastAPI (REST, JSON) backend; JavaScript frontend; Postman collection; Docker deployment.
- Contract first: the JSON shapes are frozen before building in parallel.

### Open decisions (settle in the first-week call, Section 3.1)

- Keep Kaggle only, add outside reality checks, or merge richer data (recommended: keep Kaggle for training and add outside checks, Path B).
- Which 3 to 5 extras to build (recommended list is in Section 8, with owners).
- React + Vite versus plain JavaScript (recommended: React + Vite, owned by Role 4).
- Hosting providers; final thresholds and noise model (tuned on validation data); the deadline.

### The four roles

| Role | Lane | Owns | Main deliverables |
| --- | --- | --- | --- |
| Role 1: ML Lead (team lead) | Data and models | notebooks 01-03, saved splits, chosen pipeline, class stats, arena metrics | EDA, 9-model comparison, class-wise analysis, calibration (E1-E4, E11), weather-only model, reality-check experiment, report editor |
| Role 2: Trust Engine Engineer | Trust layer | notebooks 04-05, trust\_engine.py, thresholds, noise model | OOD detectors and their evaluation, stability, verdict logic, risk profiles, explanations, conformal sets, PSI (E5-E9) |
| Role 3: Backend and DevOps Engineer | API, tests, deployment | backend/main.py, tests, postman/, Dockerfile, CI | FastAPI routes, Pydantic validation, Postman collection, pytest, Docker, live backend, logging, drift monitor endpoint (E10) |
| Role 4: Frontend and Experience Lead | Website and demo | frontend/, mock JSON, translations | React + Vite pages, what-if sliders, in-browser familiarity check, Arena and Model Card pages, mobile design, live frontend, demo and slides |

Each role is written out in full in Section 2. Names are left blank: fill in the person for each role in the tracker in Section 9.

### Planned workload (estimates, adjust after week 1)

| Role | Core work (person-days) | Extras owned (person-days) | Report and demo (person-days) | Total |
| --- | --- | --- | --- | --- |
| Role 1 | 13 | 5 | 3 | 21 |
| Role 2 | 15 | 4 | 2 | 21 |
| Role 3 | 13 | 5 | 2 | 20 |
| Role 4 | 12 | 4 | 4 | 20 |

These are planning estimates, not measurements. If one lane runs long, the lightest lane (Roles 3 and 4 at the start, Role 1 after the model is chosen) picks up overflow, and the tracker in Section 9 shows who is free.

## 2. Roles in detail

Each role below lists the mission, what you own, tasks in the order to do them, what you hand to others, what you need from others, and your report chapter. Tick the boxes in the doc as you go. Experiment numbers (E1 to E11) refer to the evaluation plan in Section 7.6.

### Role 1: ML Lead (team lead): data, models, calibration

**Mission:** produce the trained, honestly evaluated classifier and every number about it. This is the part the faculty calls most important.

**You own:** notebooks 01, 02, 03; the saved splits; crop\_pipeline.joblib; class\_stats.json; reference\_distribution.json; model\_info.json; the model-side part of arena\_metrics.json; the weather-only model; the reality-check experiment; the final report editing.

**Tasks, in order**

- [ ] Confirm the data: df.describe(), duplicates, class balance, and what period the rainfall column covers (needed before any weather API is used).
- [ ] Day 2 to 3: save stratified train / validation / test splits (suggested 70/15/15, about 1540/330/330 rows) as index files with the random seed. Commit them so Role 2 is never blocked.
- [ ] Day 3 to 4: save a quick baseline pipeline (scaler + Random Forest) as a placeholder crop\_pipeline.joblib so Roles 2 and 3 can build against a real file.
- [ ] E1, EDA: distributions, correlations, per-crop ranges and overlaps, pairwise scatter plots. Find feature pairs that never co-occur (these become the demo's impossible combinations) and write one sentence on why the dataset is easy. Write class\_stats.json (per-crop means, spreads, percentiles).
- [ ] E2, classifier comparison: Logistic Regression, KNN, SVM (RBF), Decision Tree, Random Forest, Extra Trees, Gradient Boosting or XGBoost, Gaussian Naive Bayes, small MLP. Scale for LR, KNN, SVM, MLP. Use a scikit-learn Pipeline, stratified 5-fold repeated 3 times, identical folds for every model, hyperparameters tuned inside the folds. Report mean and standard deviation of accuracy, macro-F1, log-loss, ECE, training and prediction time, and accuracy on noisy inputs. Run Wilcoxon signed-rank or McNemar and say "not significantly different" where true. Choose the final model for a stated reason.
- [ ] E3, class-wise analysis: per-crop precision, recall, F1; 22x22 confusion-matrix heatmap; a written reason for each confused pair from the feature ranges. List real rows that are misclassified: they become the demo's ambiguous inputs.
- [ ] E4, calibration: wrap the chosen model in CalibratedClassifierCV (cv=5), compare sigmoid and isotonic, keep the lower ECE and log-loss on validation. Report ECE, log-loss and Brier before and after, reliability diagrams, and the same on noisy copies of the validation set (the easy-data trap). Save the final pipeline.
- [ ] E11, ablations (bonus): accuracy without each feature; weather-only versus full model; accuracy versus training size; accuracy versus noise level per model.
- [ ] Partial-input mode: train and save a weather-only pipeline (temperature, humidity, rainfall) with the same calibration steps.
- [ ] Write arena\_metrics.json (E2, E3, E4, E11 parts), model\_info.json (versions, date, data fingerprint) and reference\_distribution.json (training histograms for drift).
- [ ] Reality-check experiment (Path B): 100 to 200 locations across India, long-term average temperature, humidity and rainfall from Open-Meteo or NASA POWER, weather-only model and gate; report the share refused, features driving refusals, and agreement with dominant regional crops (never presented as accuracy).
- [ ] Report: edit and merge all chapters; write the Model Card text with Role 4.

**You hand over:** splits (day 3), baseline pipeline (day 4), final pipeline (end of phase 1), class\_stats.json, arena\_metrics.json and reference\_distribution.json, the weather-only pipeline.

**You need:** the noise model, agreed with Role 2 in week 1; Role 2's detector class for the weather-only gate and the reality check.

**Learning goal:** you are new to APIs and deployment, so sit in on one deployment session with Role 3, run the Postman collection yourself once, and be able to answer the Part 2 viva questions.

**Report chapters:** introduction and dataset (E1), classifier comparison and class-wise analysis (E2, E3), calibration (E4), ablations (E11), reality check, limitations.

### Role 2: Trust Engine Engineer: OOD, stability, verdict, explanations

**Mission:** build the Trust Engine so the system can refuse, hedge and explain, and prove it with measurements. This is the idea that sets the project apart.

**You own:** notebooks 04, 05; backend/trust\_engine.py; ood\_stats.joblib; iforest.joblib; thresholds.json; noise\_model.json; conformal.json; the in-browser statistics file; the explanation functions; backend/tests/test\_engine.py.

**Tasks, in order**

- [ ] Week 1: with Role 3, agree the class interface: TrustEngine.load(dir) and assess(row, risk) returning exactly the JSON in Section 7.5. Publish a stub trust\_engine.py that returns the three mock replies by a trivial rule, so the API and website can be built immediately.
- [ ] Week 1: agree the noise model with Role 1 (suggested pH +/-0.3, temperature +/-1.5 C, humidity +/-5 points, N/P/K/rainfall +/-10%) and write down the reasons.
- [ ] OOD detectors, fitted on training rows only: per-crop Mahalanobis with Ledoit-Wolf shrinkage (minimum over crops), Isolation Forest, kNN-distance baseline, and the plain min/max range check as the baseline to beat. Make the fitting function take a feature list so the weather-only gate reuses it.
- [ ] E5, manufactured unknowns: leave-one-crop-out (repeat for several crops), box-uniform random inputs, shuffled-column inputs. Report AUROC and FPR at 95% TPR for all four detectors, which held-out crops the detector missed and why, and show the range check accepting impossible combinations the joint detector refuses.
- [ ] E6, thresholds: use out-of-fold scores (the 99th percentile of 330 rows rests on about 3 rows). Plot false-alarm versus detection, mark the chosen point, write thresholds.json for the three risk profiles.
- [ ] Decide the midpoint trap (an input halfway between two crops is far from both centres): consider the second gate based on kNN distance and write down how a point near two clusters is treated.
- [ ] E8, stability: noise function, 200 noisy copies in one array call, per-feature sensitivity curves, average flip rate. Show that low-stability answers are wrong more often than high-stability ones, using noisy test inputs.
- [ ] Verdict logic: decide() and trust\_score() as in Section 7.3 stage 5, three risk profiles. E7: risk-coverage curve on clean data, noisy data and manufactured unknowns.
- [ ] Explanations for POST /explain: unusual\_pair, counterfactual and smallest\_flip (decision margin), measure\_next, why-not-second-crop, comfort-zone percentiles, nearest real examples, ensemble disagreement. Word every message as relative to the nearest crop.
- [ ] E9 (bonus): conformal prediction sets, report empirical coverage and average set size.
- [ ] Export for others: ood\_browser.json (class means and inverse covariances) for Role 4; the psi() function and a log-reading helper for Role 3; the weather-only gate for Role 1.
- [ ] Unit tests for the engine so it can be tested without a web server.
- [ ] Report chapters and the evidence slides: AUROC table, reliability diagram link, risk-coverage curve.

**You hand over:** stub engine (week 1), real engine and artefacts (end of phase 1), ood\_browser.json, psi helper, weather-only gate, evidence figures.

**You need:** splits and baseline pipeline from Role 1 (day 3 to 4), final pipeline later; the frozen contract from Role 3.

**Report chapters:** trust engine design, OOD detection and evaluation (E5, E6), selective classification (E7), stability (E8), conformal (E9), pitfalls and design decisions.

### Role 3: Backend and DevOps Engineer: API, tests, deployment

**Mission:** turn the engine into a reliable, tested, deployed REST API that the website calls. Everything must work on a public URL, not just a laptop.

**You own:** backend/main.py (routes only), requirements.txt, Dockerfile, .dockerignore, backend/tests/test\_api.py, .github/workflows/tests.yml, postman/, docs/api-contract.md, the README, the repo skeleton, hosting for the backend, request logging and the drift monitor endpoint.

**Tasks, in order**

- [ ] Day 1: create the folder layout from Section 6.10, .gitignore (venv, .env, data caches), branch protection (pull request plus one review), and a README skeleton. Invite everyone.
- [ ] Day 1 to 2: write docs/api-contract.md and the three example replies (recommend, caution, refuse) with Role 4 and Role 2, then freeze them. Put them in frontend/mock/predict\_examples.json. Decide with the team whether weather-only mode is a query parameter (for example mode=weather\_only) before the freeze.
- [ ] Week 1: deploy a hello endpoint (/health) in Docker to the chosen host, fix CORS for the chosen frontend address, and create a Postman collection with a {{base\_url}} variable. Create a Postman mock server from the three example replies so Role 4 has fake data from day one.
- [ ] Routes in main.py: GET /health, GET /model-info, POST /predict?risk=, POST /predict/batch, POST /explain, GET /arena, GET /monitor. Additive extras that do not change the frozen shapes: POST /feedback and a weather helper for location autofill.
- [ ] Pydantic CropInput with physical bounds (not training min/max); field-level 422 handler; 413 for files over 1 MB; 400 for missing column or more than 500 rows; never return raw Python errors.
- [ ] Load the engine once at start-up; plain def for prediction routes; pin versions; include model\_version in replies.
- [ ] Request logging: one JSON line per request (time, verdict, OOD percentile, stability, latency, model version, the seven inputs, no personal data) into SQLite or JSON lines; rate limiting (slowapi); CORS allow-list of your own origins only.
- [ ] Tests: pytest cases for health, impossible pH (422), valid prediction, plus every row of the Postman table in Section 6.6; GitHub Actions workflow on every push.
- [ ] Postman: the full case table with test scripts, run with the Collection Runner, export the collection and environment as JSON, screenshots for the report.
- [ ] Dockerfile reading ${PORT:-8000}; run the image locally; deploy to Render, Hugging Face Spaces, Railway or Fly.io (check current free-tier rules); verify /docs on the public URL.
- [ ] Weather helper: wrap Open-Meteo or NASA POWER, with the rainfall period and units confirmed by Role 1 first; check each service's terms.
- [ ] Drift monitor: GET /monitor returns the flag rate and PSI per feature using Role 2's psi() and Role 1's reference\_distribution.json. Say plainly that free hosts may reset the disk.
- [ ] E10: median /predict time locally and on the live server.
- [ ] Final: run the Postman collection against the live URL and save the result; write the API reference and deploy steps in the README.

**You hand over:** repo skeleton and contract (day 2), Postman mock server (week 1), live hello endpoint (week 1), real API (end of phase 2), live URL with CORS correct for Role 4.

**You need:** the TrustEngine class from Role 2 (stub first), model files from Role 1, arena\_metrics.json for /arena, Role 4's final frontend address for CORS.

**Report chapters:** API design and REST rationale, validation and error handling, testing (pytest, Postman, CI), deployment, API performance (E10), monitoring, security basics.

### Role 4: Frontend and Experience Lead: website, usability, demo

**Mission:** make the trust behaviour visible and easy to use on a phone, and run the demo. The faculty named JavaScript, so this role carries the stack requirement.

**You own:** frontend/ (React + Vite recommended), the translations file, mock-driven development, frontend hosting (Vercel, Netlify or GitHub Pages), the in-browser familiarity check, the demo script, the slides and the backup video.

**Tasks, in order**

- [ ] Week 1: scaffold React + Vite (or plain JavaScript if nobody can learn React in a few days), one place for the API address (VITE\_API\_URL, baked in at build time), deploy a blank page, test it from a phone.
- [ ] Build against frontend/mock/predict\_examples.json and the Postman mock server until the real API is ready, then switch the address.
- [ ] Recommend page: seven inputs (sliders plus number boxes with typical-range hints), verdict banner (icon plus word plus colour, never colour alone), top-3 probability bars, trust meter, stability, plain-language reason in an expandable "Why?", risk-appetite switch (Conservative, Balanced, Bold), "Try typical input" and "Try impossible input" buttons, download report.
- [ ] Fetch with loading, error and "waking the server" states; AbortController plus 300 ms debounce so a stale reply never overwrites a newer one; field-level messages from 422 replies.
- [ ] What-if page: sliders that re-call the API live, counterfactual hints, and the climate stress test (call /predict twice, original and shifted temperature and rainfall, then compare; no new route needed).
- [ ] Trust displays: comfort-zone bars, "What to measure next" panel (shown on Caution), conformal set if present, nearest real examples, nutrient gap view (describe the gap only, no fertiliser quantities).
- [ ] Model Arena page (GET /arena): comparison table and charts, per-crop heatmap, reliability diagram, E5 AUROC table. Model Card page (GET /model-info): data source, tests, limits, intended use.
- [ ] Batch page: CSV upload, table of verdicts with flagged rows highlighted, downloadable report, friendly errors for 400 and 413.
- [ ] In-browser familiarity check: load ood\_browser.json from Role 2, compute Mahalanobis in JavaScript as sliders move for an instant unusual-input warning; the server stays the final authority.
- [ ] Usability: mobile-first, accessible colours, all text in one translations file (English, then Telugu and Hindi if time allows, checked by a native speaker), feedback button ("Was this wrong?"), optional PWA.
- [ ] Compare-two-fields view (two /predict calls) and season-aware ranking only if time remains and the rule table exists.
- [ ] Demo: write the script from Section 7.8 using inputs taken from Role 1's data plots, build the evidence slide deck with Role 2's figures, record a backup video and screenshots, rehearse the full run twice on a phone.
- [ ] Deploy to the host, set the API address as a build-time variable, and give Role 3 the exact origin for CORS.

**You hand over:** mock-driven site early, live site URL, demo script, slides, backup video.

**You need:** the frozen contract and mock JSON (day 2), the Postman mock server, ood\_browser.json and arena\_metrics.json, demo inputs from Role 1, evidence figures from Role 2.

**Report chapters:** frontend design and UX, accessibility and language support, in-browser check, user guide with screenshots, demo plan and conclusion, future work.

### Report and demo ownership

| Part | Writer | Reviewer |
| --- | --- | --- |
| Abstract, introduction, dataset, classifier comparison, calibration, limitations | Role 1 | Role 2 |
| Trust Engine, OOD, stability, selective classification, conformal | Role 2 | Role 1 |
| API, testing, deployment, performance, monitoring | Role 3 | Role 4 |
| Frontend, UX, user guide, demo, conclusion, future work | Role 4 | Role 3 |
| Final merge, formatting, references | Role 1 | everyone reads once |
| Slides and demo run | Role 4 (Role 2 owns the evidence slide) | Role 1 |

## 3. How we work from four places

The rule that makes remote work possible: each person edits only their own folders, and roles meet at written interfaces (files and JSON shapes), never at half-finished code. Stubs and mock data let every lane start on day one.

### 3.1 Week-1 kickoff call (one hour, everyone)

Settle these and record them in docs/decisions.md:

1. Name each person against Roles 1 to 4 (tracker in Section 9).
2. Freeze the API contract and the three mock replies (Roles 3, 4 and 2 draft it beforehand).
3. Decide whether weather-only mode is a query parameter on /predict.
4. Dataset path: A (Kaggle only), B (Kaggle plus outside checks, recommended) or C (merge); see Section 8.4.
5. Pick the 3 to 5 extras from Section 8.5 and their order.
6. React + Vite or plain JavaScript; backend and frontend hosts.
7. First draft of the noise model and the three risk profiles (placeholders until tuned).
8. Meeting rhythm and where updates are posted.

### 3.2 Folder ownership (who may edit what directly)

| Path | Owner | Others |
| --- | --- | --- |
| notebooks/01\_eda, 02\_compare\_models, 03\_calibration | Role 1 | read only, comment in pull requests |
| notebooks/04\_ood\_eval, 05\_stability | Role 2 | read only |
| backend/trust\_engine.py, backend/models/ (thresholds, noise, ood files) | Role 2 | Role 3 may import, never edit |
| backend/models/crop\_pipeline.joblib, class\_stats.json, model\_info.json, reference\_distribution.json, arena\_metrics.json | Role 1 | Role 3 reads |
| backend/main.py, tests/, requirements.txt, Dockerfile, .github/, postman/ | Role 3 | Roles 1, 2 add tests through pull requests |
| frontend/ | Role 4 | Role 3 may edit the API address only through Role 4 |
| docs/ (report, figures, decisions.md) | Role 1 edits; each writer owns their chapter file | all add figures to docs/figures/ |

### 3.3 Git rules

- One repository; main must always run. Work on short branches named role\<N>/\<topic> (for example role2/ood-detectors).
- Merge through pull requests with one reviewer from the review pair below; small commits with clear messages ("add per-crop Mahalanobis scorer").
- Never commit secrets: use environment variables and a git-ignored .env. Do not commit large data files; the 2,200-row Kaggle CSV and the split index files are small and are fine.
- Notebooks cause merge conflicts: one owner per notebook, strip outputs before committing (nbstripout), and save figures to docs/figures/ instead of relying on notebook outputs.
- Anything the API imports is a plain .py file; the API never depends on a notebook.
- Starting out for teammates new to git: git clone https://github.com/Asritha2008/crop-trust-engine, git checkout -b role1/eda, git add, git commit, git push -u origin role1/eda, then open a pull request on GitHub.

### 3.4 Review pairs

| Author | Reviewer | Why |
| --- | --- | --- |
| Role 1 | Role 2 | both ML; checks leakage rules (Section 5.6) |
| Role 2 | Role 1 | the same, in reverse |
| Role 3 | Role 4 | the contract is the meeting point |
| Role 4 | Role 3 | the same, in reverse |

Once per phase, swap: Role 1 reads one API pull request and Role 4 reads one engine pull request, so everybody can explain every layer in the viva.

### 3.5 Handoff artefacts: who produces what for whom

| Artefact | Producer | Consumer | When |
| --- | --- | --- | --- |
| docs/api-contract.md and frontend/mock/predict\_examples.json | Role 3 (with 2 and 4) | Roles 2, 4 | day 2 |
| Split index files and seed | Role 1 | Role 2 | day 3 |
| Baseline crop\_pipeline.joblib (placeholder) | Role 1 | Roles 2, 3 | day 4 |
| Stub trust\_engine.py with assess() | Role 2 | Role 3 | day 3 |
| Postman mock server | Role 3 | Role 4 | week 1 |
| Live hello endpoint plus CORS | Role 3 | Role 4 | week 1 |
| noise\_model.json | Role 2 with Role 1 | Roles 1, 2 | week 1 draft, final after E8 |
| class\_stats.json | Role 1 | Role 2 | end of EDA |
| Final crop\_pipeline.joblib | Role 1 | Roles 2, 3 | end of phase 1 |
| Real trust\_engine.py plus artefacts | Role 2 | Role 3 | end of phase 1 |
| ood\_browser.json | Role 2 | Role 4 | phase 2 |
| arena\_metrics.json (all experiments appended) | Roles 1, 2 | Roles 3, 4 | rolling |
| Weather-only pipeline and gate | Roles 1 and 2 | Role 3 | phase 2 |
| psi() helper and reference\_distribution.json | Roles 2 and 1 | Role 3 | phase 2 |
| Live API URL | Role 3 | Role 4 | end of phase 2 |
| Evidence figures | Roles 1, 2 | Role 4 (slides) | phase 3 |

### 3.6 Communication

- Daily async update in the team chat, three lines: done, doing next, blocked. Blockers are raised the same day.
- Two 30-minute video calls per week: Monday plan, Thursday demo of what works. Share screens, run the code.
- Tasks live in the checklists of this doc and in GitHub Issues or a Project board; update the tracker in Section 9 after every merge.
- Every decision that changes a threshold, a contract field or a locked decision goes in docs/decisions.md with the reason.
- Teach-back: once a week one member explains their layer for ten minutes so everyone can answer the mentor questions in Section 7.9.

### 3.7 Definition of done (every task)

- It runs from a clean checkout; the code is plain and explainable, with no unexplained snippets.
- It has a test or a saved output, and is merged through a reviewed pull request.
- No leakage: everything that learns from data is fitted on training or validation data; the test set is touched once at the end.
- Every extra feature comes with one measurable claim and one slide of evidence.
- Nothing is reported as a result until it has been produced by our own code on our own data; numbers in this handbook are placeholders.

## 4. Timeline and dependencies

The deadline is not fixed, so phases use relative effort. Phase 1 has four independent lanes; the only waits are the day-2 to day-4 handoffs in Section 3.5. Overall effort share follows the guide: step 0 about 5%, EDA 10%, comparison 15%, trust engine 25%, API 15%, frontend 15%, deployment 5%, extras 5%, report and buffer 5% plus buffer.

| Phase | Role 1 (ML) | Role 2 (Trust) | Role 3 (Backend) | Role 4 (Frontend) | Gate to pass |
| --- | --- | --- | --- | --- | --- |
| 0. Week 1 setup (about 5%) | Data checks, splits (day 3), baseline pipeline (day 4), noise-model draft | Class interface plus stub engine, noise-model draft | Repo skeleton, contract, mocks, hello endpoint deployed, Postman mock server | Scaffold, blank page deployed, mock JSON wired | Contract frozen; hello endpoint live; CORS fixed; all four have run everything locally |
| 1. Build in parallel (about 55%) | E1 to E4, class\_stats, weather-only model | Detectors, E5 to E8, verdict logic, explanation functions | Routes, validation, tests, Postman cases, Docker, logging, CI | Recommend page, what-if, Arena, Model Card, Batch against mocks | Final pipeline and real engine files committed; API passes tests against the stub; pages work on mocks |
| 2. Integrate (about 15%) | Arena metrics, reality-check experiment, ablations | ood\_browser.json, psi helper, weather-only gate, conformal | Plug real engine in, /explain, /arena, /monitor, live deployment | Switch to live API, in-browser check, charts, language file | End-to-end run from a phone on the public URL; Postman collection passes live |
| 3. Extras and polish (about 10%) | Model Card text, extra analyses | Evidence figures, edge cases, extras logic | E10, hardening, README | Compare, stress test, UI polish, feedback button | Every extra has one claim and one evidence slide |
| 4. Report, rehearsal, buffer (about 10%) | Merge and edit the report | Write chapters, evidence slide | Write chapters, final live Postman run | Slides, backup video, two rehearsals | Report complete; demo rehearsed twice; backup video recorded |

If any lane falls behind, cut in this order: conformal sets, batch CSV, drift monitor, Telugu and Hindi. Never cut the must-haves: comparison, class-wise report, calibration, stability, working OOD flag with three verdicts, API, website, deployment.

## 5. Part 1: The essentials every member must own

Read this first (about 30 minutes). Sub-numbers match the original guide (5.1 is guide 1.1, and so on). Calibration (5.4) is written in full because it is the concept teams most often get wrong.

### 5.1 What we are building

A user enters seven numbers about a field: Nitrogen, Phosphorus, Potassium (N, P, K), soil pH, temperature, humidity and rainfall. A model trained on 2,200 labelled rows (22 crops, 100 rows each) says which crop fits. That part is a textbook exercise and every group will reach about 99% accuracy, because the dataset is clean and each crop sits in its own tight region of the 7-number space. The marks are in the other half of the problem: knowing when the answer should not be trusted, and wrapping everything in a working, deployed web app.

### 5.2 The problem statement, line by line

| The statement says | What it is really asking | What we build |
| --- | --- | --- |
| Build a crop classifier from soil and weather variables | Supervised classification: 7 numbers in, 1 of 22 labels out | A tuned, calibrated model returning the top 3 crops with honest probabilities |
| Compare classifiers | A fair experiment, not a list of accuracies: same splits for every model, several metrics, evidence the differences are real | 9 models, stratified repeated k-fold; accuracy, macro-F1, log-loss, calibration error, speed; a paired significance test |
| Measure class-wise performance | Per-crop precision, recall, F1; which crops get confused and why | Confusion-matrix heatmap and a written reason for each confused pair, from the feature ranges |
| Test recommendation stability under input changes | Sensitivity analysis: real soil tests and weather figures have errors; does the answer survive them? | A noise model, per-feature sensitivity curves, and a stability % on every answer |
| Flag combinations unusual relative to training data | Out-of-distribution (OOD) detection; the key word is combinations: every value can be normal while the pair is not | A multivariate detector (per-crop Mahalanobis plus Isolation Forest), explained and tested |
| Instead of confidently recommending a crop for every input | The system must be able to abstain | Three verdicts: Recommend, Recommend with caution, Refuse, each with a reason |

### 5.3 Idea one: OOD, and why a normal classifier cannot do it

OOD (out-of-distribution) means the input comes from a region of the input space the training data never covered. A classifier is forced to choose among its 22 crops, so it will pick one even for garbage, and often with high confidence. Confidence measures how clearly the input falls on one side of a boundary between known crops. It does not measure whether the input is anywhere near known data.

**Real-life example:** a fruit-sorting machine trained on apples, oranges and bananas is shown a brick. It does not say "I do not know"; it says "orange, 91%", because the brick happens to lie on the orange side of a boundary. An OOD detector is a second component asking a different question: is this input close to any known cluster?

**Why a per-feature min/max check fails.** Suppose in training, high humidity (about 90%) always came with high rainfall (above about 150 mm). An input of humidity 92% and rainfall 25 mm has both numbers inside their individual min-max ranges, so the range check passes it, yet that pair never occurred. Only a detector that looks at all seven numbers together can catch it. That is why the statement says "combinations".

**Two different reasons to doubt an answer.** Most teams merge these. We keep them apart:

|  | Familiar input | Unfamiliar input (OOD) |
| --- | --- | --- |
| Clear winner | Recommend: normal input, one crop clearly ahead | Refuse: the dangerous case, a confident answer about something the model never saw |
| Two crops close | Caution: normal input but ambiguous; show the top 3 and say why | Refuse: unfamiliar and the model is unsure as well |

### 5.4 Idea two: calibration (making confidence honest)

**What it is.** A model outputs a probability for each crop, for example rice 0.93. Its confidence is the highest of these (0.93). The model is calibrated if confidence matches reality: among all predictions where it says about 0.90, about 90% should turn out right. A weather app that says "70% chance of rain" is calibrated if it rains on about 70 of 100 such days. If it rains on only 45, the app is over-confident.

**Real-life example:** take 100 predictions where the model said "about 90% sure". If 92 are correct, the model is honest. If only 78 are correct, it is over-confident, and every "90%" shown to a user overstates the truth by 12 points. Accuracy alone would never reveal this, because accuracy does not look at what the model claimed.

**Why it matters in this project**

- We show percentages to users. A displayed "93%" is a promise; if it is not true, the interface is misleading.
- Our verdict rules use thresholds on probability (caution if top probability is below 0.70, or the margin between the top two is below 0.20). Those numbers only mean something if probabilities are honest.
- The trust score, the conformal sets (5.8) and the risk-appetite slider (Section 8.3) all consume probabilities. Bad probabilities contaminate all of them.
- It is in the title: "Model Confidence". A mentor will ask how you know your confidence can be trusted.

**Why models are often not calibrated.** Accuracy training does not reward honest probabilities. Random Forest averages the votes of many trees, which tends to keep probabilities away from the extremes. Naive Bayes and boosted trees tend to be over-confident; an SVM outputs scores that are not probabilities until converted; neural networks are often over-confident. Which direction each model errs in is something we measure rather than assume.

**How to read a reliability diagram.** Group predictions into confidence bins (0.0-0.1, 0.1-0.2, ... 0.9-1.0). For each bin, plot the average confidence on the x-axis and the fraction actually correct on the y-axis. A perfectly calibrated model lies on the diagonal. A curve below the diagonal means over-confident (says 90%, delivers 78%); a curve above means under-confident. Draw our own from our experiments before and after calibration.

**How to measure it.** Expected Calibration Error (ECE): split predictions into confidence bins; in each bin compare average confidence with actual accuracy; ECE is the size-weighted average of those gaps. Lower is better. Worked example (illustrative numbers, 500 predictions):

| Confidence bin | Predictions | Average confidence | Actual accuracy | Gap | Gap x share |
| --- | --- | --- | --- | --- | --- |
| 0.6 - 0.7 | 50 | 0.66 | 0.58 | 0.08 | 0.08 x 0.10 = 0.008 |
| 0.7 - 0.8 | 80 | 0.75 | 0.70 | 0.05 | 0.05 x 0.16 = 0.008 |
| 0.8 - 0.9 | 120 | 0.86 | 0.79 | 0.07 | 0.07 x 0.24 = 0.017 |
| 0.9 - 1.0 | 250 | 0.97 | 0.91 | 0.06 | 0.06 x 0.50 = 0.030 |
| Total | 500 |  |  |  | ECE = 0.063 (6.3 points) |

Also report log-loss (punishes confident mistakes) and the Brier score (mean squared error of the probabilities); both fall when probabilities improve.

```python
def reliability_bins(probs, y_true, classes, bins=10):
    top = probs.max(axis=1)                      # the model's confidence per prediction
    pred = classes[probs.argmax(axis=1)]
    correct = (pred == y_true)
    n = len(y_true)
    ece = 0.0
    points = []                                  # (avg confidence, accuracy) per bin -> plot these
    for b in range(bins):
        in_bin = (top > b / bins) & (top <= (b + 1) / bins)
        if in_bin.sum() > 0:
            conf = top[in_bin].mean()
            acc = correct[in_bin].mean()
            points.append((conf, acc))
            ece = ece + (in_bin.sum() / n) * abs(acc - conf)   # weight each bin by its size
    return ece, points
```

**How to fix it**

| Method | What it does | When to use |
| --- | --- | --- |
| Platt / sigmoid scaling | Fits a small logistic curve that maps the model's scores to probabilities | Little data; smooth, stable correction |
| Isotonic regression | Fits a free-form, monotonic step function mapping scores to probabilities | Enough data (a few hundred per fold); more flexible but can overfit small samples |
| Temperature scaling | Divides the model's raw scores by one learned number T before the softmax | Neural-network models |

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

base = RandomForestClassifier(n_estimators=300, random_state=42)
# cv=5: the calibrator is fitted on held-out folds INSIDE the training data, so the
# validation and test sets stay untouched for thresholds and honest evaluation
model = CalibratedClassifierCV(base, method="isotonic", cv=5).fit(X_train, y_train)
probs = model.predict_proba(X_val)               # calibrated probabilities
```

- In scikit-learn the multiclass case is handled one-versus-rest per crop and then renormalised. Try method="sigmoid" and "isotonic" and keep the one with lower ECE and log-loss on validation data.
- Calibration data must be separate from the data used to score the result. The cross-validated wrapper above does this; never calibrate and report on the same rows.

**Pitfalls**

1. **The easy-data trap.** At about 99% accuracy almost every prediction falls in the top bin, so ECE is tiny and the reliability diagram looks dull. Evaluate calibration also on noisy copies of the validation set (add the noise model from 5.5): errors appear and miscalibration becomes visible.
2. **Calibration is not accuracy.** It changes how probabilities are expressed, not which crop wins.
3. **Calibration does not detect unfamiliar inputs.** A perfectly calibrated model is still confident about a brick-like input, because calibration only fixes honesty on inputs like the training data. That is why the OOD gate exists and runs before the classifier.

**Mentor question: "Why not just show predict\_proba?"** Answer: raw probabilities from tree models are not calibrated. We measure ECE before and after calibration, show a reliability diagram, and test it on noisy inputs, so our percentages mean what they claim; and we pair it with an OOD gate because calibration alone cannot detect unfamiliar inputs.

### 5.5 Idea three: stability (sensitivity analysis)

A soil lab can be off by 0.3 pH; a weather figure by a few percent. If a 0.2 pH change flips the answer from rice to jute, the input sits on a decision boundary and the answer should not be presented as firm. Method: define a noise model (for example pH +/-0.3, temperature +/-1.5 C, humidity +/-5 points, N/P/K/rainfall +/-10%), create about 200 noisy copies of the input, predict each, and report the share that keep the same top crop. That share is the stability score. Also compute per-feature sensitivity curves (vary one feature at a time) and the average flip rate over the test set.

**Confidence is not stability.** Confidence is the model's own probability for one exact input. Stability asks whether the answer survives small realistic changes. A prediction can be 95% confident and still fragile if it sits right next to a boundary. Mentors like this distinction.

### 5.6 One rule that protects every number you report

Data leakage means information from the test data sneaks into training or tuning, inflating results (for example fitting a scaler on the full dataset before splitting, or choosing a threshold on the test set). Rule: split first; fit the scaler, model, calibrator, OOD detector and thresholds on training or validation data only; touch the test set once. A scikit-learn Pipeline does this automatically inside cross-validation.

| Component | Fit on | Evaluate on |
| --- | --- | --- |
| Scaler and classifier | Training split (inside a Pipeline; refit in each CV fold) | Validation, then test once |
| Hyperparameters | Cross-validation on the training split | Never the test set |
| Calibrator | Held-out folds inside training (cv=5) | Validation, then test |
| OOD detector | Training split | Validation rows (for thresholds); manufactured unknowns; test |
| Thresholds (percentiles, 0.70, ...) | Validation split, ideally out-of-fold scores | Test set, once, at the end |

### 5.7 Comparing classifiers properly

- **Candidates:** Logistic Regression, KNN, SVM (RBF), Decision Tree, Random Forest, Extra Trees, Gradient Boosting or XGBoost, Gaussian Naive Bayes, small MLP. Scale features for LR, KNN, SVM and MLP; trees do not need it.
- **Protocol:** stratified 5-fold repeated 3 times, identical folds for every model, hyperparameters tuned inside the folds, mean and standard deviation reported.
- **Metrics:** accuracy, macro-F1, log-loss, ECE, training and prediction time; also accuracy on noisy inputs.
- **Significance:** with 99% accuracy most differences are noise. Run a paired test across folds (Wilcoxon signed-rank) or McNemar's test, and say "not significantly different" when true. Choose the final model for a stated reason (for example, accuracy ties, so lowest log-loss and ECE).

### 5.8 Advanced ideas that lift the project

| Idea | What it is |
| --- | --- |
| Selective classification | A system that may abstain trades coverage (share of inputs it answers) against risk (error on the answered ones). Sort test predictions from most to least trusted and plot accuracy at each coverage level: the risk-coverage curve. It proves refusing buys reliability and is the basis of the risk-appetite slider (Conservative, Balanced, Bold = different thresholds). On clean data the curve is flat; draw it on noisy inputs and manufactured unknowns. |
| Conformal prediction | Return a set of crops guaranteed (on average) to contain the true crop with a frequency you choose, such as 95%. One crop when sure, several when not. It assumes new inputs resemble the calibration data, which is exactly why the OOD gate runs first; the two methods complement each other. |
| Explanations that act | Counterfactual: the smallest change that would alter the verdict. Decision margin in real units: "rainfall can drop 62 mm before the answer changes". What to measure next: when two crops compete, name the single input that best separates them, so re-measuring it resolves the doubt. |
| Drift monitoring | Real inputs may stop resembling the training data. The Population Stability Index (PSI) compares a live feature's distribution with training (below 0.1 stable; above 0.25 a significant shift, a common rule of thumb); the flag rate is the share of requests refused. A monitor page turns "we know when to distrust the model" into an ongoing capability. |

## 6. Part 2: The software stack

Everything between the saved model file and the user's screen: what each piece is, why you need it, how to use it, and how it fails. Sub-numbers match the original guide (6.3 is guide 2.3).

### 6.1 The journey of one click

The model never talks to the user directly. It lives in a Python program on a server; the user sits in a browser. In between, the two sides exchange short text messages over the internet. One click: the browser (user clicks Recommend) calls fetch() which sends JSON to FastAPI, which checks the input and runs the engine using the model files (pipeline, OOD statistics); FastAPI returns a JSON reply, and the page draws the verdict.

| Piece | Restaurant picture | Our technology |
| --- | --- | --- |
| The trained model and OOD detector, saved as files | The chef | scikit-learn, joblib |
| The program that receives requests and returns answers | The waiter and the menu | FastAPI, run by Uvicorn |
| The page the user sees | The dining room | JavaScript (React or plain), HTML, CSS |
| The code in the page that calls the server | The customer placing an order | Fetch API |
| Testing the server without any page | A mystery shopper | Postman |
| Putting it on the public internet | Opening on the street | Docker plus Render / Hugging Face Spaces plus Vercel |

The browser and the server only agree on the shape of the JSON messages (the API contract, Section 7.5), so the website can be built while the model is still being trained. This is how four people work at the same time.

### 6.2 HTTP and JSON: the language of the web

HTTP is the set of rules browsers and servers follow when they talk. A conversation is one request followed by one response. A request has four parts: a method (what to do), a URL (where), headers (extra information, such as "my body is JSON") and an optional body (the data). A response has a status code, headers and a body.

```
https://crop-api.onrender.com/predict?top=3
\___/ \____________________/ \______/ \____/
scheme   host (server)        path      query (optional extras)
```

| Method | Meaning | In our project |
| --- | --- | --- |
| GET | Read something. No body | GET /health, GET /model-info, GET /arena |
| POST | Send data for the server to process | POST /predict with the 7 numbers in the body |
| PUT / DELETE | Replace / remove a stored thing | Not needed in this project |

| Status | Meaning | When we send it |
| --- | --- | --- |
| 200 OK | It worked | Every prediction, including a Refuse verdict. A refusal is a correct answer from a working server, not an error |
| 400 | Malformed request | A broken CSV or missing column in a batch upload |
| 404 | No such address | A typo in the URL |
| 413 / 429 | Too large / too many requests | File over the size limit; rate limit hit |
| 422 | Well-formed but invalid data | FastAPI sends this automatically for pH = 40 or a missing field |
| 500 | Our code crashed | A bug. Read the server log |

JSON is the text format both sides use for data. A JavaScript object becomes JSON with JSON.stringify; JSON becomes an object again with response.json().

```json
// Request body (browser -> server)
{ "N": 90, "P": 42, "K": 43, "temperature": 21.0, "humidity": 82.0, "ph": 6.5, "rainfall": 203.0 }

// Response body (server -> browser)
{ "verdict": "recommend",
  "top_crops": [ {"crop": "rice", "probability": 0.91}, {"crop": "jute", "probability": 0.05} ],
  "ood": { "flagged": false, "percentile": 41.0 },
  "stability": 0.94, "trust_score": 88,
  "explanation": "Inputs match known rice conditions and are stable under small changes." }
```

### 6.3 API, REST, FastAPI, Fetch, Postman: what each one actually is

These words sit at different levels; mixing them up is the most common viva mistake.

| Term | What kind of thing | Where it runs | Its job for us |
| --- | --- | --- | --- |
| API | A concept: an agreed way for one program to ask another to do something | Everywhere | The umbrella idea. Our API is the set of addresses the website can call |
| REST | A design style for web APIs: addresses stand for things, HTTP methods say the action, data is JSON, every request stands alone (stateless) | A set of rules, not software | The rules our API follows |
| FastAPI | A Python library for building a REST API | Server | We write routes like /predict in it |
| Uvicorn | The server program that runs a FastAPI app and listens for requests | Server | Started with one command; FastAPI is the car body, Uvicorn the engine |
| Pydantic | A Python library FastAPI uses to define and check the shape of incoming data | Server | Rejects pH = 40 before the model sees it |
| OpenAPI / Swagger | A standard description of an API; FastAPI generates it and serves an interactive page at /docs | Server | Free documentation, a live demo tool, and a file Postman can import |
| Fetch API | A function built into every browser for calling an API | Browser | How our page sends the 7 numbers and receives the verdict |
| Axios | A JavaScript library doing the same job as fetch with conveniences | Browser | Optional. We use fetch |
| Postman | A desktop app for sending requests and checking replies, with no page involved | Your laptop | Tests the backend early and is the proof that you tested it |
| CORS | A browser security rule about which websites may call which servers | Enforced by the browser | Must be configured on the server (6.8) |

In one line: we design the API in the REST style, build it on the server with FastAPI (run by Uvicorn, validated by Pydantic), call it from the browser with Fetch, and test it with Postman.

**Other API styles and why we do not use them**

| Style | How it works | Choose it when | Why not us |
| --- | --- | --- | --- |
| REST | Many addresses, HTTP methods, JSON | Almost any web app | Our choice. Simple, stateless, trivial to test in Postman |
| GraphQL | One address; the client states exactly which fields it wants | Many related data types, many different clients | We have one prediction call; overkill |
| WebSocket | A connection that stays open for live two-way messages | Chat, live scores | Nothing here is live-streamed |
| gRPC | Fast binary calls between services | Microservices inside a company | Browsers cannot call it directly |
| SOAP | Strict XML messages | Old banking and enterprise systems | Heavy and outdated |

**Mentor: why FastAPI and REST?** Our model is Python, so a Python framework avoids a translation layer. FastAPI validates inputs automatically through Pydantic, produces interactive documentation at /docs, and is fast. REST is stateless: each /predict call carries all seven numbers, so the server needs no memory of earlier requests, which makes it simple to scale and test. Flask would also work but needs manual validation and documentation.

**Mentor: "The stack says JavaScript and ML."** In our design JavaScript is the whole frontend and runs a small familiarity check in the browser. The ML runs in Python on the server. If asked whether the model could run in the browser: yes in principle (ONNX Runtime Web or TensorFlow.js), at the cost of converting models; we keep it on the server and say so.

### 6.4 Building the API with FastAPI

```bash
python -m venv venv                  # isolated dependencies
venv\Scripts\activate                # Windows. Mac/Linux: source venv/bin/activate
pip install fastapi uvicorn scikit-learn joblib numpy pandas python-multipart
pip install pytest httpx             # tests (the test client needs httpx)
pip freeze > requirements.txt        # exact versions: essential for deployment
uvicorn main:app --reload            # then open http://127.0.0.1:8000/docs
```

A complete minimal backend. Route functions stay tiny; the thinking lives in a separate trust\_engine.py (Section 7), so the engine can be tested without a web server.

```python
# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from trust_engine import TrustEngine          # our own module

app = FastAPI(title="Crop Trust Engine", version="1.0.0")
app.add_middleware(CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://your-site.vercel.app"],
    allow_methods=["*"], allow_headers=["*"])

engine = TrustEngine.load("models/")          # loaded ONCE, when the server starts

class CropInput(BaseModel):                   # Pydantic checks every request
    N: float = Field(ge=0, le=200)
    P: float = Field(ge=0, le=200)
    K: float = Field(ge=0, le=300)
    temperature: float = Field(ge=-10, le=60)
    humidity: float = Field(ge=0, le=100)
    ph: float = Field(ge=0, le=14)
    rainfall: float = Field(ge=0, le=1000)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: CropInput, risk: str = "balanced"):
    return engine.assess(data.model_dump(), risk)   # Pydantic v1 used .dict()
```

- **Validation is free.** pH = 40 or a missing rainfall gets a 422 with a clear message and the model is never called. Keep two cases apart: invalid input (pH 40, physically impossible) is an error; valid but unusual input (each value plausible, the combination never seen) is an OOD case and returns 200 with verdict Refuse. Choose Field limits as physical bounds, not training min/max.
- **Load the model once** at start-up, not inside the route. Loading on every request makes the API very slow.
- **Use plain def, not async def,** for prediction routes. Inference is CPU work; FastAPI runs plain functions in worker threads so the server stays responsive.
- **Save the whole pipeline** (scaler + model + calibrator) as one joblib file, and the OOD statistics and thresholds as a second file. If the API preprocesses differently from training, answers silently go wrong (training-serving skew).
- **Pin versions.** A model saved with one scikit-learn version can fail under another. Deploy with the same requirements.txt.
- **Version the API** (title and version string, optionally a /v1 prefix) so the frontend knows what it is talking to; include a model\_version field in replies.

**Batch upload and friendly errors**

```python
import io
import pandas as pd
from fastapi import UploadFile, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

@app.post("/predict/batch")
async def predict_batch(file: UploadFile, risk: str = "balanced"):
    raw = await file.read()
    if len(raw) > 1_000_000:
        raise HTTPException(status_code=413, detail="File too large (limit 1 MB)")
    df = pd.read_csv(io.BytesIO(raw))
    for col in FEATURES:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail="Missing column: " + col)
    if len(df) > 500:
        raise HTTPException(status_code=400, detail="At most 500 rows per upload")
    results = []
    for i in range(len(df)):
        row = {}
        for col in FEATURES:
            row[col] = float(df.iloc[i][col])
        results.append(engine.assess(row, risk))
    return {"count": len(results), "results": results}

@app.exception_handler(RequestValidationError)      # field-level messages for the website
async def friendly_errors(request, exc):
    problems = []
    for err in exc.errors():
        problems.append({"field": str(err["loc"][-1]), "message": err["msg"]})
    return JSONResponse(status_code=422, content={"errors": problems})
```

Uploads need python-multipart, and always cap file size and row count so a big upload cannot crash the server during the demo.

### 6.5 Calling the API from JavaScript: the Fetch API

fetch sends the request and returns a promise: a placeholder for a reply that has not arrived yet. async/await lets you write the waiting as top-to-bottom code.

```javascript
const API = import.meta.env.VITE_API_URL;   // Vite project: one place to change

async function getRecommendation(values, risk, signal) {
  showLoading(true);
  try {
    const res = await fetch(API + "/predict?risk=" + risk, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(values),          // JS object -> JSON text
      signal: signal                         // lets us cancel this request
    });
    if (!res.ok) {                           // 422, 500, ...
      const err = await res.json();
      throw new Error("Server said: " + JSON.stringify(err.errors || err.detail));
    }
    showResult(await res.json());            // JSON text -> JS object -> draw it
  } catch (e) {
    if (e.name === "AbortError") return;     // we cancelled it ourselves
    showError(e.message);                    // network down, server asleep, bad input
  } finally {
    showLoading(false);
  }
}

// What-if sliders: wait until the user pauses, and cancel any older request
let timer = null;
let controller = null;
function onSliderChange(values, risk) {
  clearTimeout(timer);
  timer = setTimeout(function () {
    if (controller) controller.abort();      // stale reply must not overwrite a newer one
    controller = new AbortController();
    getRecommendation(values, risk, controller.signal);
  }, 300);
}
```

- **Always show loading and error states.** Free hosts can take up to a minute to wake after sitting idle; without a "waking the server" message the site looks broken in front of the judges.
- **Keep the API address in one environment variable.** import.meta.env works only in a Vite project, and Vite bakes the value in at build time, so change it and rebuild. In plain JavaScript use a constant at the top of one file. Avoid a silent fallback to localhost in production: a missing variable should fail loudly.
- **Debounce sliders.** Without it, dragging fires dozens of requests per second.

### 6.6 Testing with Postman, pytest and CI

Postman lets you call your API before any website exists and proves afterwards that it behaves. The mentor mentioned it specifically, so treat the collection as a deliverable: screenshots and the exported file go in the report.

1. Create a Collection "Crop Trust Engine". Add a variable {{base\_url}} (local: http://127.0.0.1:8000; live: your Render URL). You can also import /openapi.json to generate the requests.
2. Add a request: POST {{base\_url}}/predict; Body tab, raw, JSON; paste a sample input; Send.
3. Add tests (a script that runs after each response), then run the whole collection with the Collection Runner.
4. Use a Postman mock server to give the website team example replies on day one, before the backend exists (contract-first).
5. Export the collection and environment as JSON and commit them. Optionally run them in CI with the Newman command-line runner.

```javascript
pm.test("Status is 200", () => pm.response.to.have.status(200));
pm.test("Reply has a verdict", () => {
  pm.expect(pm.response.json()).to.have.property("verdict");
});
pm.test("Refusal gives a reason", () => {   // only in the 'impossible combination' request
  const b = pm.response.json();
  pm.expect(b.verdict).to.eql("refuse");
  pm.expect(b.explanation).to.be.a("string");
});
```

| Test case | Expected |
| --- | --- |
| Typical row for one crop, taken from your data | 200, verdict recommend, plausible top crop |
| A real row your confusion matrix shows is misclassified (not an average of two crops, see Section 7.4) | 200, verdict caution, top 3 shown |
| Impossible combination (each value in range, pair never seen) | 200, verdict refuse, reason given |
| pH = 40 | 422 validation error naming the field |
| Missing rainfall field | 422 validation error |
| Same input with risk=conservative versus bold | Verdict may differ; probabilities identical |
| Batch with 3 rows; batch with a missing column | 200 with 3 results; 400 naming the column |
| GET /health | 200, status ok |

Add the same cases as automated Python tests so they run on every change:

```python
# backend/tests/test_api.py      run with: pytest   (needs: pip install pytest httpx)
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)
GOOD = dict(N=90, P=42, K=43, temperature=21, humidity=82, ph=6.5, rainfall=203)

def test_health():
    assert client.get("/health").status_code == 200

def test_rejects_impossible_ph():
    bad = dict(GOOD)
    bad["ph"] = 40
    assert client.post("/predict", json=bad).status_code == 422

def test_returns_verdict():
    assert "verdict" in client.post("/predict", json=GOOD).json()
```

Continuous integration (CI) runs the tests automatically on every push, so a teammate's change cannot silently break the API before the demo. A GitHub Actions workflow is a few lines:

```yaml
# .github/workflows/tests.yml
name: tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r backend/requirements.txt pytest httpx
      - run: cd backend && pytest tests
```

### 6.7 The frontend

Recommendation: React + Vite if at least one teammate can learn it in a few days: components keep the several pages tidy and chart libraries plug in cleanly. Plain HTML/CSS/JavaScript is a valid fallback with fewer moving parts, but gets messy past two pages. Either satisfies the "JavaScript" requirement. Use Chart.js or Recharts for bars, the reliability diagram and the heatmap, and design phone-first: judges and farmers will open the link on one.

| Page | What it shows |
| --- | --- |
| Recommend | 7 inputs (sliders plus number boxes, with typical-range hints); verdict banner; top-3 probability bars; trust meter; stability; plain-language reason; risk-appetite switch |
| What-if | Sliders that re-call the API live so users see the verdict and probabilities move; counterfactual hints; climate stress test |
| Model Arena | Classifier comparison table and charts; per-crop heatmap; reliability diagram. Fed by GET /arena |
| Batch | Upload a CSV, get a table of verdicts, download a report with flagged rows highlighted |
| Model Card | Data source, how it was tested, known limits, intended use. Fed by GET /model-info |

**Recommend page layout (wireframe in words).** A top bar with the tabs Recommend, What-if, Arena, Batch, Model card and a language switch (EN | TE | HI). Left: the seven inputs with typical-range hints, plus "Try typical input" and "Try impossible input" buttons. Right: the verdict banner (for example "RECOMMEND: RICE"), the top 3 with calibrated probabilities (Rice 91%, Jute 5%, Maize 2%), a trust meter and stability (TRUST 88, STABILITY 94%), the risk-appetite switch (Conservative | Balanced | Bold), an expandable "Why?" (inputs match known rice conditions; stable under realistic sensor noise; decision margin such as "rainfall can drop by 62 mm before the answer changes"), a "What to measure next" box shown on Caution (for example "Rice vs jute is decided mostly by K; re-test potassium"), and buttons for Climate stress test and Download report.

Principles: the verdict comes first (icon plus word plus colour, never colour alone), numbers second, reasons third (an expandable "Why?"); always say what to do next; show the typical range beside every input; include "try a typical input" and "try an impossible input" buttons for the demo.

### 6.8 CORS: the bug everyone hits

When the website (for example on vercel.app) calls the API (on onrender.com), the browser sees two different origins. For safety it blocks the call unless the server replies "this website is allowed". Symptom: the request works in Postman but the browser console says "blocked by CORS policy". Cause of the confusion: Postman does not enforce CORS, so it makes the server look fine. Fix: list the website's exact address in allow\_origins (6.4) and redeploy the backend. While debugging only, \["\*"\] allows everything; do not leave it in the final version.

### 6.9 Docker and deployment

The problem Docker solves: your laptop has particular Python and library versions; the hosting server does not. Docker lets you describe the whole environment in a short Dockerfile, builds a portable image from it, and runs the image as a container. If it runs in the container on your laptop, it runs the same online.

```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]

# .dockerignore:  venv/  __pycache__/  .git/  notebooks/  .env
# run locally:    docker build -t crop-api .   then   docker run -p 8000:8000 crop-api
```

The ${PORT:-8000} part matters: many hosts tell your app which port to use through an environment variable; some expect a fixed port, so read their docs.

| Piece | Where it can live | Notes (check each provider's current rules) |
| --- | --- | --- |
| Backend (FastAPI + model) | Render, Hugging Face Spaces (Docker), Railway, Fly.io | Free tiers often sleep when idle; the first request can take a minute. Open the site just before the demo |
| Frontend | Vercel, Netlify, GitHub Pages | Set the API address as an environment variable at build time |
| Model files | Inside the image or the repo | Small files; no separate storage needed |
| Request logs (drift page) | SQLite file or JSON lines | Free hosts may reset the disk on redeploy. State the limit or use a small hosted database |

**Deploy a trivial version in week one.** Deploy a "hello" endpoint early; hosting problems (ports, versions, memory, CORS) are discovered in an hour instead of the night before the demo. Then: (1) Postman passes locally; (2) the Docker image runs locally; (3) push to GitHub; (4) create the backend service from the repo, test /health and /docs on the public URL; (5) set the frontend's API address and the backend's CORS list, deploy the frontend, test from a phone; (6) run the Postman collection against the public URL and save the result.

### 6.10 Repository layout and working together

```
crop-trust-engine/
|-- notebooks/   01_eda  02_compare_models  03_calibration  04_ood_eval  05_stability
|-- backend/
|   |-- main.py            routes only
|   |-- trust_engine.py    OOD gate, calibration, stability, verdict logic (no web code)
|   |-- models/            crop_pipeline.joblib  ood_stats.joblib  thresholds.json
|   |-- tests/test_api.py  requirements.txt  Dockerfile
|-- frontend/              src/ (pages)  package.json  mock/predict_examples.json
|-- postman/               crop-trust-engine.postman_collection.json
|-- docs/                  report, figures      README.md
```

- **Notebooks are for experiments; plain .py files are for anything the API imports.** The API must never depend on a notebook.
- **Git:** one repository, one short branch per feature, merge through pull requests, small commits with clear messages. Never commit secrets (use environment variables and a git-ignored .env). Do not commit large data files.
- **Contract first:** freeze the JSON shapes (7.5) and put three example replies (recommend, caution, refuse) in frontend/mock/. The frontend builds against that file while the backend is unfinished.
- **README:** what it is, how to run locally, how to deploy, API reference, live link. Mentors read it first.
- **Logging and basic safety:** log one JSON line per request (time, verdict, OOD percentile, stability, latency, model version, the seven inputs; no personal data) to feed the monitor page. Add rate limiting (for example the slowapi library), size caps on uploads, a CORS allow-list of your own origins, and never return raw Python errors to users.

### 6.11 What breaks, and how to recognise it

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Works in Postman, fails in the browser with "CORS policy" | Website origin not allowed on the server | Add the exact origin to allow\_origins; redeploy |
| 422 on every request | Field names or types do not match the Pydantic model (for example "pH" vs "ph") | Read the error detail; match names exactly |
| Live site calls localhost | API address hard-coded, or env variable set after the build | Use the environment variable; rebuild the frontend |
| Browser blocks the call on the live site | https site calling an http API (mixed content) | Use the https URL of the backend |
| Server crashes on start in Docker | Model path wrong, or scikit-learn version differs from training | Use paths relative to the file; pin versions; rebuild |
| Upload endpoint fails at start-up | python-multipart missing | Add it to requirements.txt |
| Tests fail with an httpx error | httpx not installed | pip install httpx |
| First request takes a minute | Free host was asleep | Show a loading message; open the site before the demo |
| Slow predictions | Model loaded inside the route; or 200 noisy copies predicted in a Python loop | Load once; predict all noisy rows in one array call |
| Answers differ from the notebook | Training-serving skew: different preprocessing | Save and load the complete pipeline |

## 7. Part 3: Our design

The architecture we propose, the reasoning behind each choice, the code skeleton of the trust layer, the API contract, the evaluation plan and the advanced components. Sub-numbers match the original guide (7.3 is guide 3.3). All code is a starting point: type it, run it, break it, and understand each line.

### 7.1 Design choices and the reason for each

| Choice | Why |
| --- | --- |
| Focus on trust and refusal, not raw accuracy | The dataset is easy, so every group reports about 99%. The last two sentences of the statement (stability, unusual combinations) are where depth is rewarded |
| Keep low confidence and OOD separate | Different failures need different responses. Ambiguity: show alternatives. Unfamiliarity: do not answer |
| Joint (multivariate) detector, not range checks | The statement says combinations |
| Gate first, classifier second | Calibration and conformal guarantees only hold for inputs that resemble the training data, so unfamiliar inputs must be stopped before they reach the classifier's probabilities |
| Evaluate the detector with manufactured unknowns | Without a test, "our detector works" is only a claim. Hardly anyone does this |
| Separate API from website; contract first | Parallel work, a clean contract, testable in Postman, reusable by other clients |
| Explain every verdict and say what to do next | A refusal without a reason frustrates users. A reason builds trust and shows understanding |
| State limitations openly | The data is idealised. Honest scope reads as maturity |

### 7.2 Architecture: the Trust Engine

Five stages in a line: (1) Validate: impossible values rejected with 422. (2) OOD gate: has the system seen anything like this? (3) Calibrated classifier: honest top 3. (4) Stability: perturb and re-predict. (5) Verdict: recommend, caution or refuse, plus a reason.

The Trust Engine is one Python class (trust\_engine.py) with a method assess(row, risk) that returns the whole JSON reply. The API only calls it. This keeps the intelligence testable without a web server and lets notebooks and the API use identical code. Training writes files ("artefacts"); the API only reads them.

| Artefact file | Written by | Contains |
| --- | --- | --- |
| crop\_pipeline.joblib | notebooks 02/03 (Role 1) | Scaler + chosen classifier + calibrator, as one pipeline |
| ood\_stats.joblib, iforest.joblib | notebook 04 (Role 2) | Per-crop mean, covariance and inverse covariance; Isolation Forest |
| thresholds.json | notebook 04 (Role 2) | Refuse and caution cut-offs for each risk profile (stage 5) |
| noise\_model.json | notebook 05 (Role 2 with Role 1) | Per-feature noise sizes and physical LOW/HIGH limits |
| conformal.json (bonus) | notebook 03/04 (Role 2) | alpha and the quantile q |
| class\_stats.json | notebook 01 (Role 1) | Per-crop means and spreads, for explanations and "what to measure next" |
| arena\_metrics.json | notebooks 02-05 (Roles 1, 2) | Results of E2 to E8 for the Arena page and /arena |
| model\_info.json, reference\_distribution.json | all (Role 1) | Versions, date, data fingerprint; training histograms for PSI drift |

### 7.3 The five stages in detail

**Stage 1: validation.** Done by Pydantic limits in the route (6.4). Choose limits as physical bounds (pH 0 to 14, humidity 0 to 100), not as training min/max; being outside the training range is the OOD gate's job.

**Stage 2: the OOD gate.** Mahalanobis distance measures how far a point is from the centre of a cloud of data, in units that account for the spread of each feature and the correlation between features. Real-life picture: a person 190 cm tall is normal on a basketball team and unusual in a primary-school class; "unusual" always means relative to a group. Because our data is 22 separate clusters, one global distance is meaningless, so we compute the distance to each crop's own cloud and take the smallest. With only 100 rows per crop, use a shrinkage covariance (Ledoit-Wolf) so the matrix is stable.

```python
import numpy as np
from sklearn.covariance import LedoitWolf

stats = {}                                    # fit on TRAINING rows only
for crop in crops:
    rows = X_train[y_train == crop]
    cov = LedoitWolf().fit(rows).covariance_
    stats[crop] = (rows.mean(axis=0), np.linalg.inv(cov), cov)

def ood_score(x):
    best_d2 = None
    best_crop = None
    for crop in stats:
        mean, inv_cov, cov = stats[crop]
        diff = x - mean
        d2 = float(diff @ inv_cov @ diff)     # squared distance to this crop's cloud
        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best_crop = crop
    return best_d2, best_crop                 # small = familiar, large = unusual

# Threshold: score VALIDATION rows (better: out-of-fold scores), then take a high percentile
val_scores = np.array([ood_score(x)[0] for x in X_val])
refuse_at = np.percentile(val_scores, 99)     # about 1% of genuine inputs falsely refused
caution_at = np.percentile(val_scores, 95)
```

Second detector: Isolation Forest. It splits the data at random; unusual points get isolated in very few splits and score as outliers. It needs no distribution assumption but its score is harder to explain. Also compute a kNN-distance baseline (distance to the k-th nearest training row). Report all three plus the min/max range check and choose the final detector from the evidence. The threshold trade-off is real: lowering the percentile catches more unusual inputs but falsely flags more normal ones; plot that curve and mark your choice. Percentiles from a small validation set are noisy (the 99th percentile of 330 rows rests on about 3 rows), so prefer out-of-fold scores from cross-validation.

Explaining the flag. To say which combination is unusual, compute the distance for each pair of features against the nearest crop, and report the pair that is most unusual while both single values look normal:

```python
from itertools import combinations

def unusual_pair(x, mean, cov, names, single_z=3.0):
    sd = np.sqrt(np.diag(cov))
    best = None
    for i, j in combinations(range(len(x)), 2):
        zi = (x[i] - mean[i]) / sd[i]
        zj = (x[j] - mean[j]) / sd[j]
        if abs(zi) > single_z or abs(zj) > single_z:
            continue                           # a single value is already odd; skip
        d = np.array([x[i] - mean[i], x[j] - mean[j]])
        sub = cov[np.ix_([i, j], [i, j])]
        d2 = float(d @ np.linalg.inv(sub) @ d)
        if best is None or d2 > best[0]:
            best = (d2, names[i], names[j])
    return best                                # e.g. (31.2, "humidity", "rainfall")
```

**Stage 3: the calibrated classifier.** The calibrated pipeline from 5.4 gives honest probabilities. Also compute the margin (top probability minus second): a small margin means two crops are close, which is the "caution" case. Isotonic calibration needs enough data; if data is scarce use sigmoid (Platt) and say why.

```python
probs = model.predict_proba(x_row.reshape(1, -1))[0]
order = np.argsort(-probs)                     # indices from most to least likely
top_prob = probs[order[0]]
margin = probs[order[0]] - probs[order[1]]
top3 = []
for k in range(3):
    top3.append({"crop": str(model.classes_[order[k]]), "probability": round(float(probs[order[k]]), 3)})
```

**Stage 4: the stability check.**

```python
def stability(predict_fn, x, sigma, n=200, seed=0):
    rng = np.random.default_rng(seed)
    noisy = x + rng.normal(0, sigma, size=(n, len(x)))   # n noisy copies at once
    noisy = np.clip(noisy, LOW, HIGH)                    # stay physically valid
    base = predict_fn(x.reshape(1, -1))[0]
    return float((predict_fn(noisy) == base).mean())     # 1.0 solid, 0.6 fragile

# sigma = per-feature noise sizes, e.g. [10, 10, 10, 1.5, 5, 0.3, 0.10 * rainfall]
```

One predict call on all noisy rows, not 200 separate calls. The same function powers the climate stress test (shift temperature and rainfall, re-run the engine) in 7.10.

**Stage 5: the verdict and the risk-appetite profiles.**

```python
def decide(ood_score, top_prob, margin, stab, profile):
    if ood_score > profile["refuse_at"]:
        return "refuse"
    if (ood_score > profile["caution_at"] or top_prob < profile["min_prob"]
            or margin < profile["min_margin"] or stab < profile["min_stability"]):
        return "caution"
    return "recommend"

def trust_score(verdict, top_prob, stab):      # simple and documented, NOT a probability
    if verdict == "refuse":
        return 0
    score = 100 * min(top_prob, stab)
    if verdict == "caution" and score > 60:
        score = 60
    return round(score)
```

| Risk profile (user switch) | Refuse above OOD percentile | Min top probability | Min margin | Min stability |
| --- | --- | --- | --- | --- |
| Conservative | 97th | 0.80 | 0.30 | 0.90 |
| Balanced (default) | 99th | 0.70 | 0.20 | 0.80 |
| Bold | 99.5th | 0.55 | 0.10 | 0.65 |

These are starting values, not facts. The percentiles, 0.70, 0.20 and 0.80 are sensible first guesses. Tune them on validation data, show the trade-off curve, and report the final values and why.

### 7.4 Pitfalls to design around now

**The midpoint trap.** A per-crop distance measures closeness to a crop's own cluster. An input exactly halfway between two crops is far from both centres, so it may be flagged Refuse instead of Caution. A demo "ambiguous input" made by averaging two crops could therefore go wrong. Handle it deliberately: (a) pick ambiguous demo inputs from real rows that the confusion matrix shows are misclassified, not from averages; (b) consider a second gate based on kNN distance, which is low anywhere near real training rows; (c) decide how a point near two clusters should be treated and write it down. A good design discussion for Roles 1 and 2.

- Do not train the OOD detector on unusual examples. It learns normal data only; unusual data is for testing it.
- Clusters this tight make everything look odd. The dataset is idealised, so real-world inputs may be flagged often. This is a finding to report, not a failure.
- Do not tune on the test set. Thresholds come from validation data; the test set is used once.
- Pair explanations are relative to the nearest crop, not a universal truth. Word the message accordingly.

### 7.5 The API contract

Agree this first and freeze it (Roles 3, 4 and 2). Everything else can then be built in parallel against it.

| Method and path | Purpose | Notes |
| --- | --- | --- |
| GET /health | Is the server alive? | Used by the host and by Postman |
| GET /model-info | Model name, versions, thresholds, training metrics | Feeds the Model Card page |
| POST /predict?risk=balanced | Full Trust Engine result for one input | The main route; risk is conservative, balanced or bold |
| POST /predict/batch | Same for many rows (CSV upload) | One result per row plus a summary; capped size |
| POST /explain | Unusual pair, counterfactuals, decision margins, feature contributions | Heavier; called on demand |
| GET /arena | Classifier comparison and per-crop metrics as JSON | Feeds the Arena charts |
| GET /monitor (bonus) | Flag rate and drift indicators | Feeds a monitoring page |

A refusal reply. Note it is still HTTP 200:

```json
{ "verdict": "refuse",
  "top_crops": [],
  "ood": { "flagged": true, "percentile": 99.8,
           "unusual_pair": ["humidity", "rainfall"], "nearest_known": ["rice", "jute"] },
  "stability": null, "trust_score": 0, "model_version": "2026-11-07-a",
  "explanation": "Humidity 92% and rainfall 25 mm are each common, but this combination
                  does not appear in the training data. No recommendation is made." }
```

A caution reply (numbers are illustrative):

```json
{ "verdict": "caution",
  "top_crops": [ {"crop": "rice", "probability": 0.52}, {"crop": "jute", "probability": 0.41},
                 {"crop": "maize", "probability": 0.04} ],
  "conformal_set": ["rice", "jute"],
  "ood": { "flagged": false, "percentile": 63.0 },
  "stability": 0.71, "trust_score": 52,
  "measure_next": "Potassium (K): it separates rice from jute most strongly. Re-test it.",
  "explanation": "Two crops are close and a small change in the inputs can flip the answer.",
  "model_version": "2026-11-07-a" }
```

The recommend reply is the example in 6.2. Role 3 stores all three in frontend/mock/predict\_examples.json.

### 7.6 Evaluation plan (this becomes the results chapter)

| # | Experiment | Output | Owner |
| --- | --- | --- | --- |
| E1 | EDA: distributions, correlations, per-crop ranges and overlaps | Figures; a sentence on why the dataset is easy | Role 1 |
| E2 | Classifier comparison: 9 models, repeated stratified k-fold, tuned | Table with mean and std for accuracy, macro-F1, log-loss, ECE, times; significance notes | Role 1 |
| E3 | Class-wise analysis of the chosen model | Per-crop report, 22x22 heatmap, explained confusable pairs | Role 1 |
| E4 | Calibration before and after (also on noisy inputs) | ECE numbers, reliability diagrams | Role 1 |
| E5 | OOD detectors compared: range check, per-class Mahalanobis, Isolation Forest, kNN distance | AUROC and FPR at 95% TPR on three kinds of manufactured unknowns | Role 2 |
| E6 | Threshold choice | False-alarm versus detection curve, chosen point marked | Role 2 |
| E7 | Selective classification | Risk-coverage curve: error should fall as the system refuses more | Role 2 |
| E8 | Stability analysis | Per-feature sensitivity curves; flip rate; do low-stability answers err more often? | Role 2 |
| E9 | Conformal prediction sets (bonus) | Empirical coverage and average set size | Role 2 |
| E10 | API performance | Median /predict time locally and on the live server | Role 3 |
| E11 | Ablations (bonus) | Accuracy without each feature; weather-only versus full; accuracy versus training size; accuracy versus noise level per model | Role 1 |

**How to manufacture "unknowns" for E5** (you have no real unusual data, so you create it):

- **Leave-one-crop-out:** remove one crop (say coffee) from training entirely, train the detector on the other 21, then check that coffee rows are flagged while held-out rows of the 21 known crops pass. Repeat for several crops. This simulates a crop the system has never met.
- **Box-uniform random inputs:** random points inside each feature's min-max range. The range check passes all of them; a joint detector should flag most.
- **Shuffled-column inputs:** take real rows and shuffle each column independently. Every value is realistic but the relationships are broken.

AUROC (0.5 = coin flip, 1.0 = perfect) is the probability that a randomly chosen unusual input gets a higher detector score than a randomly chosen normal one. FPR at 95% TPR says how many normal inputs are wrongly flagged when you catch 95% of the unusual ones. Risk-coverage: coverage is the share of inputs still answered; as it drops, error on the answered ones should drop.

**Two results that make the project convincing.** (1) In E8, show that low-stability or low-trust predictions really are wrong more often than high-trust ones (use noisy test inputs so errors exist); that proves the trust score carries information. (2) In E5, show the plain range check accepting the impossible combinations that the joint detector refuses. Also report which held-out crops the detector missed and why; a detector that never fails invites suspicion.

### 7.7 Build order for four people

Only the order that avoids people waiting on each other. Relative effort, because the deadline is unknown.

| Step | Work | Result | Share | Lead |
| --- | --- | --- | --- | --- |
| 0 | Repo, environments, freeze the API contract, mock JSON, deploy a "hello" endpoint | Contract plus live hello | 5% | Role 3 (all join) |
| 1 | EDA and splits | Notebook 01, saved splits | 10% | Role 1 |
| 2 | Compare classifiers; class-wise analysis | Notebook 02; chosen model; metrics JSON | 15% | Role 1 |
| 3 | Calibration, OOD detectors and their evaluation, stability, verdict logic | trust\_engine.py plus saved artefacts | 25% | Role 1 (calibration) and Role 2 (rest) |
| 4 (parallel) | FastAPI routes, tests, Postman collection, Docker | API running locally | 15% | Role 3 |
| 5 (parallel) | Frontend against mock JSON, then the real API | Site running locally | 15% | Role 4 |
| 6 | Deploy both; CORS; end-to-end test from a phone; Postman on the live URL | Live link | 5% | Roles 3 and 4 |
| 7 | Pick standout features (Section 8) | Extras | 5% | all, per Section 8.5 owners |
| 8 | Report, rehearsal, buffer | Final deliverables | 5% plus buffer | all, Role 1 merges |

### 7.8 Demo plan

- **Normal input:** typical conditions for one crop. Recommend, high trust, stable. Take values from your own data; do not trust numbers copied from a guide.
- **Ambiguous input:** a real row that your confusion matrix shows is misclassified. Caution with top 3, a reason and "what to measure next"; drag a slider and watch probabilities shift; flip the risk switch to show the verdict change.
- **Impossible combination:** pick a pair that your scatter plots show never occurs (for example high humidity with very low rainfall) while each value is individually in range. Refuse with the explanation, then one slide showing the range check accepting the same input.
- **Evidence slide:** AUROC table, reliability diagram, risk-coverage curve.
- **Engineering:** the /docs page and a Postman run, in one minute. Keep screenshots or a short video as a backup in case the internet or the host fails.

### 7.9 Limitations, and questions a mentor may ask

- The dataset is clean and idealised (100 rows per crop, tight ranges), so results will look better than field data would produce.
- Only seven variables are used: no soil type, season, prices, pests or water availability. "Refuse" means "outside what this dataset taught the model", not "this land is bad".
- The output is decision support, not agronomic advice. Thresholds and noise levels are our choices, with stated reasoning.

| Question | Short answer |
| --- | --- |
| The dataset gives 99%. What is your contribution? | Reliability: calibrated confidence, a combination-aware detector evaluated by leave-one-crop-out, a stability score, explained refusals |
| Low confidence vs OOD? | Low confidence: familiar input, two crops close; we show alternatives. OOD: unfamiliar input, the model has no basis to answer even if it sounds sure; we refuse |
| Why not min/max checks? | They miss combinations. Random points inside the min-max box all pass; our joint detector flags most. We show the numbers |
| How do you know the detector works? | We manufacture unknowns (leave-one-crop-out, box-uniform, shuffled columns), report AUROC against baselines, and report where it fails |
| Why not just show predict\_proba? | Tree-model probabilities are not calibrated. We measure ECE before and after, test on noisy inputs, and still gate OOD because calibration cannot detect unfamiliar inputs (5.4) |
| Why does a refusal return 200? | It is a correct answer from a working server, not a request or server error. Invalid values such as pH 40 return 422 |
| How did you avoid leakage? | Everything that learns from data is fitted on training or validation data inside a pipeline; the test set is used once (5.6) |
| Why compare nine models if one gets 99%? | The task asks for it; models differ in log-loss, calibration, speed and stability even when accuracy ties |
| Why is the gate before the classifier? | Calibration and conformal guarantees assume inputs resemble the training data; the gate stops inputs that do not |
| Could the model run in the browser? | Yes in principle (ONNX Runtime Web, TensorFlow.js); we keep the model on the server and run only a small familiarity check in JavaScript |

### 7.10 Advanced components: code

**Leave-one-crop-out detector test**

```python
from sklearn.metrics import roc_auc_score

auroc = {}
for held_out in crop_names:
    keep = (y_train != held_out)
    detector = fit_detector(X_train[keep], y_train[keep])       # never sees the held-out crop
    normal = score_all(detector, X_val[y_val != held_out])      # scores of known crops
    unknown = score_all(detector, X_all[y_all == held_out])     # all 100 rows of the unseen crop
    labels = [0] * len(normal) + [1] * len(unknown)             # 1 = unusual
    auroc[held_out] = roc_auc_score(labels, list(normal) + list(unknown))
```

**Risk-coverage curve**

```python
def risk_coverage(trust, correct):            # lists: trust score and 1/0 correctness per row
    order = np.argsort(-np.array(trust))      # most trusted first
    correct = np.array(correct)[order]
    curve = []
    for k in range(10, len(correct) + 1, 10): # answer the k most trusted inputs
        curve.append((k / len(correct), correct[:k].mean()))    # (coverage, accuracy)
    return curve
```

**Conformal prediction set**

```python
import math

cal_scores = []
for i in range(len(y_cal)):
    cal_scores.append(1 - cal_probs[i][class_index[y_cal[i]]])  # low = model agreed with the truth
n = len(cal_scores)
level = min(math.ceil((n + 1) * (1 - alpha)) / n, 1.0)           # finite-sample correction
q = np.quantile(cal_scores, level)

def predict_set(probs):
    chosen = []
    for k in range(len(probs)):
        if 1 - probs[k] <= q:
            chosen.append(classes[k])
    return chosen
```

**What to measure next, decision margin, climate stress test**

```python
def measure_next(crop_a, crop_b, class_means, class_sds, names):
    best_name = None
    best_gap = 0.0
    for j in range(len(names)):
        pooled = np.sqrt((class_sds[crop_a][j] ** 2 + class_sds[crop_b][j] ** 2) / 2)
        gap = abs(class_means[crop_a][j] - class_means[crop_b][j]) / pooled
        if gap > best_gap:
            best_gap = gap
            best_name = names[j]
    return best_name        # the feature that separates the two crops most

def smallest_flip(model, x, feature_index, step, max_steps=60):
    base = model.predict([x])[0]
    for k in range(1, max_steps + 1):
        for direction in (1, -1):
            trial = list(x)
            trial[feature_index] = x[feature_index] + direction * k * step
            if model.predict([trial])[0] != base and is_familiar(trial):   # realistic AND different
                return direction * k * step     # the decision margin in real units
    return None

def stress_test(engine, row, delta_temp, rain_factor):
    shifted = dict(row)
    shifted["temperature"] = row["temperature"] + delta_temp
    shifted["rainfall"] = row["rainfall"] * rain_factor
    return engine.assess(shifted, "balanced")   # compare with the original verdict
```

**Drift: Population Stability Index**

```python
def psi(train_vals, live_vals, bins=10):
    edges = np.quantile(train_vals, np.linspace(0, 1, bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    t = np.histogram(train_vals, bins=edges)[0] / len(train_vals)
    l = np.histogram(live_vals, bins=edges)[0] / len(live_vals)
    t = np.clip(t, 1e-6, None)                 # avoid log(0)
    l = np.clip(l, 1e-6, None)
    total = 0.0
    for i in range(bins):
        total = total + (l[i] - t[i]) * np.log(l[i] / t[i])
    return total                               # below 0.1 stable, above 0.25 shift
```

## 8. Part 4: Features and inputs to stand out

A menu of ideas to choose from. Sub-numbers match the original guide (8.5 is guide 4.5). The recommended picks and their owners are in 8.5.

### 8.1 How to choose

Judges see many projects with the same 99% model. What they remember is a feature that makes them think "this person understood the real problem". Score every idea on four questions: Wow (will a judge remember it?), Effort (days for one person), Data (does it need information we do not have?) and Risk (can it embarrass us live?). Prefer high wow, low effort, no new data.

**Order of work.** Finish the must-haves first: comparison, class-wise report, stability, working OOD flag with three verdicts, API, website, deployment. Extras on top of a broken base lose marks; extras on top of a solid base win them. Every extra should come with one measurable claim and one slide of evidence.

### 8.2 Extra inputs the user could provide

The seven dataset features are only part of what a real farmer knows. Extra inputs can be used in two ways: (a) as a rule layer applied after the model (filter or re-rank the top crops), which needs no retraining; (b) as new model features, which needs a dataset that contains them. Option (a) is realistic for us. Rule tables must be hand-built from cited agricultural sources and labelled as rules, not learned results.

| Input | Why it helps | How we would use it | Effort |
| --- | --- | --- | --- |
| Location (state/district or map pin) | Farmers rarely know today's humidity or seasonal rainfall; location lets the app fetch weather itself | Call a free weather API (Open-Meteo or NASA POWER) and pre-fill temperature, humidity, rainfall. Check each service's terms and the period its rainfall covers | Medium |
| Season (kharif / rabi / zaid) or sowing month | A crop can fit the numbers but not the calendar | A hand-built table of crop versus sowing season; filter or down-rank crops outside the season | Low |
| Soil type (alluvial, black, red, sandy, loamy) | Farmers know their soil type even without a lab test | Suitability table used to re-rank the top crops; shown as a reason ("loamy soil suits this crop") | Low-Med |
| Previous crop | Crop rotation matters; repeating a crop or following certain families harms yield | Simple rotation rules (for example legumes before cereals); warn on repeats | Low |
| Water availability (rain-fed / irrigated) | Rainfall alone misstates water for irrigated land | If irrigated, treat the rainfall input as rain plus irrigation, or reduce the weight of the rainfall flag; show it as an assumption | Low |
| Missing / unsure inputs | Many farmers have no soil test, so N, P, K, pH are unknown | Partial-input mode: train a second model on weather only (temperature, humidity, rainfall) and show it with visibly lower trust. Also accept a range for any input and report the share of the range that gives the same crop | Medium |
| Risk appetite (Conservative / Balanced / Bold) | Users differ in how much doubt they accept | A switch mapped to the threshold profiles in 7.3, stage 5 | Low |
| Land size, budget | Turns a recommendation into a decision | Only with extra cost and price data; otherwise skip | High (data) |
| Soil-test report upload | No typing; works from a photo or PDF of a soil health card | OCR to extract N, P, K, pH; user confirms the numbers. Risky (OCR errors, varied layouts) | High |
| Altitude | Affects temperature and crop choice | Fetch from an elevation API once location is known | Low |

### 8.3 Feature ideas

**A. Trust and explanation (our core, so go deep here)**

| Feature | What the user sees | How it works | Wow | Effort |
| --- | --- | --- | --- | --- |
| Why was this flagged? | "Humidity 92% and rainfall 25 mm are each common but never occur together" | Pairwise Mahalanobis distance (code in 7.3) | High | Low |
| What would change this? (counterfactual) | "Raising rainfall from 25 to 120 mm would make this input recognisable; the recommendation would be chickpea" | For each feature, step its value up and down until the top crop changes or the flag clears; report the smallest change. Say it is about the model, not agronomy | High | Medium |
| Decision margin in real units | "Rainfall can drop 62 mm before the answer changes" | The same search, reported per feature as a safety margin (7.10) | High | Low |
| What to measure next | "Rice and jute are close; re-test potassium, it separates them most" | Largest standardised gap between the top two crops' means (7.10). Shown only on Caution | High | Low |
| Risk-appetite slider | Conservative / Balanced / Bold changes how readily the system refuses, with the risk-coverage curve behind it | Threshold profiles (7.3); selective classification (5.8) | High | Low |
| Climate stress test | "If rainfall drops 30% and temperature rises 2 C, the recommendation moves from rice to cotton and trust falls to 55" | Shift the inputs by a scenario and re-run the engine; reuses the stability code | High | Low |
| Comfort-zone bars | For each input, a bar showing where the value sits inside the recommended crop's typical range | Per-crop percentiles from the training data; draw a marker | High | Low |
| Why not the second crop? | Top 3 with the feature that separates them ("jute needs more rainfall than you entered") | Compare the input with each crop's mean; report the largest standardised differences | Medium | Low |
| Nearest real examples | Three training rows most similar to the input | kNN lookup; show them in a table | Medium | Low |
| Conformal prediction set | A set of crops that contains the true one with a chosen guarantee (say 95%). One crop when sure; several when not | Code in 7.10. Report coverage and average set size | High | Medium |
| Ensemble disagreement | A flag when RF, SVM and KNN disagree | Compare the top crops of three models; disagreement is a second uncertainty signal | Medium | Low |

**B. Decision support for the farmer**

| Feature | What it does | Note | Wow | Effort |
| --- | --- | --- | --- | --- |
| Nutrient gap view | Compares the user's N, P, K and pH with the typical range for the recommended crop and says low / ok / high | Describe the gap only. Do not give fertiliser quantities unless you can cite an agronomic source | Medium | Low |
| Season-aware ranking | Re-ranks results using the season input | Needs the crop-season table (8.2) | Medium | Low |
| Compare two fields | Two input panels side by side; verdicts and probabilities shown together | Two calls to /predict | Medium | Low |
| Yield or profit estimate | Rough expected return per crop | Needs yield and price data we do not have. Only if you switch or add a dataset | High | High |

**C. Access and usability**

| Feature | What it does | Note | Wow | Effort |
| --- | --- | --- | --- | --- |
| Mobile-first design | Layout that works one-handed on a phone | Cheap and expected; judges will open it on a phone | Medium | Low |
| Telugu / Hindi interface | Labels and explanations in the local language | Keep all text in one translations file; machine drafts need a native speaker's check | Medium | Medium |
| Voice input | Say the numbers aloud | Browser speech APIs vary by browser and language; treat as a bonus demo only | Medium | Medium |
| In-browser familiarity check | Instant "unusual input" warning as sliders move, with no server call | Export class means and inverse covariances as JSON; compute Mahalanobis in JavaScript. The server stays the final authority. This also makes the JavaScript + ML stack real | High | Medium |
| Offline / installable (PWA) | Works with weak connectivity | Only the in-browser check works fully offline; the full model stays on the server | Medium | Medium |

**D. Engineering and ML in production**

| Feature | What it does | Note | Wow | Effort |
| --- | --- | --- | --- | --- |
| Model Arena page | Live comparison table, per-crop heatmap, reliability diagram | Mostly reads saved JSON from /arena | High | Low-Med |
| Batch CSV upload with report | Verdict for every row; flagged rows highlighted; downloadable | Natural use for agriculture offices and extension workers | Medium | Medium |
| Feedback button | "Was this wrong?" with the correct crop; stored with the inputs | Shows human-in-the-loop retraining thinking. Store in a small database | Medium | Low-Med |
| Drift monitor | Share of requests flagged over time; PSI of incoming features against training (code in 7.10) | Needs request logging; free hosts may reset disks | High | Medium |
| Reality-check mode | How often the system refuses real-world climates (8.4) | A finding about the data, not a feature of the product alone | High | Medium |
| Model card page | Intended use, data, tests, limits | Cheap, and mentors respect it | Medium | Low |
| CI and tests | pytest and Postman run on every push | A GitHub Actions workflow is a few lines (6.6) | Medium | Low |

### 8.4 About changing the dataset

You can change the dataset but do not want to decide now. Here is the thinking to have ready.

| Path | What it means | When it makes sense |
| --- | --- | --- |
| A. Keep Kaggle | Use the given dataset; be honest that it is idealised | Default. Everything in this handbook is designed for it |
| B. Keep it, add outside checks | Keep Kaggle for training. Use a second source (real weather from an API, published crop-region information) as an extra test of how the OOD gate behaves on real-world values | Best value for effort. It turns the "idealised data" weakness into a finding |
| C. Merge or replace | Train on a richer dataset, ideally with soil type, season, region or yield | Only after the must-haves work. Changes every downstream result and threshold |

**Checklist for judging any new dataset:** Are its features measured or synthetic? How many rows per crop, and how many crops? Is it labelled with the same kind of target (best crop, not yield)? Does it include the new inputs we want (season, soil type, region)? Is the licence open? Can you explain where it came from to a mentor? Also check how the Kaggle set defines its units, especially rainfall (which period it covers), before connecting any weather API, or the numbers will not be comparable. The Kaggle page describes the data as built by augmenting Indian rainfall, climate and fertiliser data; read that description and be ready to say so.

**Sources worth investigating** (described from general knowledge; verify availability, licence and limits yourself)

| Source | What it offers | Use in our project | Caution |
| --- | --- | --- | --- |
| FAO EcoCrop | Published environmental requirement ranges per crop (temperature, rainfall, pH ...) | Cross-check the ranges our model learned; build the season and suitability rule layer | Coarse ranges; check licence |
| NASA POWER | Climate and weather by latitude and longitude, via an API | Auto-fill temperature, humidity, rainfall; reality check | Resolution is coarse; confirm the rainfall period and units |
| Open-Meteo | Free weather and historical-weather API | A simpler alternative for the same job | Read the terms of use |
| ISRIC SoilGrids | Global soil property maps (pH, nitrogen and more) via an API | Pre-fill pH as a suggestion | Units and depth differ from the Kaggle N, P, K; do not feed them in as-is |
| data.gov.in, Soil Health Card portal | Indian open government data: regional crop and soil statistics | Region-level sanity checks | Formats vary; check coverage |
| ICRISAT district-level database | Which crops are actually grown, by Indian district | Soft validation: does our recommendation agree with the dominant local crops? | Reflects farmer choice and markets, not suitability alone |
| Agmarknet (market prices) | Mandi price data | Price-aware ranking, bonus only | Needs care; do not claim profit |

**The reality-check experiment (Path B, recommended; owner Role 1)**

1. Choose 100 to 200 locations across India (for example district centres).
2. Fetch long-term average temperature, humidity and rainfall for each from a weather source. Decide which rainfall period matches the Kaggle rainfall column and write the decision down.
3. Soil is unknown here, so use the weather-only model (temperature, humidity, rainfall) and a weather-only OOD gate trained on the same Kaggle rows.
4. Report: the share of locations refused, which features drive the refusals, and how often the recommendation agrees with the dominant crops in the regional statistics.
5. Present it as a finding: how a model trained on idealised data behaves on real climates, and why a trust layer is necessary. Do not present agreement as accuracy.

Why it is worth the effort: it gives the OOD component a real-world reason to exist, it is honest about the dataset's limits, and almost nobody will have it.

**Integration patterns for new inputs and data**

| Pattern | How | Good for | Limit |
| --- | --- | --- | --- |
| Auto-fill | Location becomes the weather features through an API call | Location, altitude | Unit and period mismatches |
| Rule layer | Hand-built, cited tables filter or re-weight the model's top crops | Season, soil type, rotation, irrigation | Rules are not learned; label them as such |
| Second model | Train a smaller model on the inputs available | Weather-only mode | Lower accuracy; show lower trust honestly |
| Retrain | Add the new features to a richer dataset and redo everything | Only with real labelled data | Changes all results and thresholds |

### 8.5 Recommended picks for a team of four, with owners

Ordered by wow per day of effort. Adjust after you see how long the must-haves take.

| Pick | Feature | Why this one | Logic | API | Screen |
| --- | --- | --- | --- | --- | --- |
| 1 | Why flagged + what would change this + decision margin | Directly answers the statement's core. Few groups will explain a refusal | Role 2 | Role 3 (/explain) | Role 4 |
| 2 | Risk-appetite switch + what to measure next | Cheap, interactive, and rare: it turns thresholds and statistics into something the user controls and acts on | Role 2 | Role 3 (risk parameter) | Role 4 |
| 3 | Climate stress test | Memorable demo; reuses the stability engine; fits the "stability" line of the statement | Role 2 (stress\_test, optional) | none needed (two /predict calls) | Role 4 |
| 4 | In-browser familiarity check with live sliders | Makes the JavaScript + ML stack real; instant and visual | Role 2 (exports ood\_browser.json) | none | Role 4 |
| 5 | Partial-input mode (weather only, or ranges) | Solves a real farmer problem (no soil test) and shows trust dropping honestly | Roles 1 (model) and 2 (gate) | Role 3 | Role 4 |
| 6 | Location autofill + reality-check experiment | Strongest "real app" feel and an honest finding about the data. Verify rainfall units first | Role 1 (experiment) | Role 3 (weather helper) | Role 4 |
| 7 | Model Arena + Model Card + feedback button | Cheap, polished, shows ML-ops thinking | Role 1 (metrics) | Role 3 (/arena, /model-info, /feedback) | Role 4 |
| If time | Conformal sets, batch CSV, drift monitor, Telugu/Hindi | Impressive but each needs care; add one or two, not all four | Role 2 / Role 3 / Role 3 / Role 4 |  |  |

**The story to tell the mentor:** "Our system does not just answer; it tells you how much to trust the answer, why, what would change it, and what to measure next, and we tested every one of those claims." Whatever extras you pick, make each tell that story.

**Skip list.** Soil-report OCR, voice input, yield and profit estimates, and a WhatsApp bot look exciting but carry the highest risk of breaking in front of judges, or need data you do not have. Mention them under "future work".

## 9. Verify, first-week checklists and progress tracker

This handbook is a plan. No code in it has been run against the team's data, and every number that looks like a result is a placeholder or an illustration. Treat it as a map, not as finished work.

### 9.1 Verify before relying on anything

- **Code is untested.** Every snippet is a starting point. Type it, run it, break it, and understand each line.
- **Dataset facts.** Row count, classes and ranges are as published on Kaggle; confirm yours with df.describe() and check for duplicates.
- **Thresholds and noise levels** (99th/95th percentile, 0.70, 0.20, 0.80, the noise sizes, the risk profiles) are placeholders. Tune on validation data and report the final values.
- **Numbers that look like results are illustrations** (the ECE worked example, the sample probabilities, the reliability-diagram shapes). Replace them with your own.
- **Hosting rules change.** Check the current free-tier behaviour of Render, Hugging Face Spaces and Vercel before depending on one; free backends often sleep when idle.
- **Weather and soil APIs and datasets:** read each service's current terms, limits and the period its rainfall covers.
- **Library versions.** Examples assume Pydantic v2 (model\_dump), recent scikit-learn and FastAPI. Pin what you actually install.
- **Demo inputs** must come from your own data and plots, not from any guide.
- **Workload estimates** in Section 1 are planning guesses; revisit them after week 1.

### 9.2 First-week checklist, by person

**Everyone**

- [ ] Clone the repository, create a virtual environment, and run the hello endpoint locally.
- [ ] Attend the kickoff call (Section 3.1) and write the decisions into docs/decisions.md.
- [ ] Read Section 5 (about 30 minutes) and be able to explain OOD, calibration and stability in your own words.

**Role 1**

- [ ] df.describe(), duplicate check, class balance, rainfall-period check.
- [ ] Commit stratified train/validation/test split indices and the seed (day 3).
- [ ] Commit a baseline pipeline (day 4).
- [ ] Plot pairs of features; list pairs that never co-occur (future demo inputs).
- [ ] Agree the noise model with Role 2 and write it down with reasons.

**Role 2**

- [ ] Agree the TrustEngine interface with Role 3 and commit the stub (day 3).
- [ ] Draft the noise model and the three risk profiles as placeholders.
- [ ] Prototype the per-crop Mahalanobis scorer on the baseline split.

**Role 3**

- [ ] Create the folder layout, .gitignore, README skeleton, branch protection.
- [ ] Write docs/api-contract.md and the three mock replies, then freeze them.
- [ ] Deploy the hello endpoint in Docker; fix CORS; create the Postman collection and mock server.

**Role 4**

- [ ] Scaffold React + Vite (or plain JavaScript) and deploy a blank page; test it on a phone.
- [ ] Wire the Recommend page to frontend/mock/predict\_examples.json.
- [ ] Draft the page list and wireframes from Section 6.7.

### 9.3 Progress tracker (fill in as you go)

| Component | Owner | Status | Notes |
| --- | --- | --- | --- |
| API contract frozen + mock JSON | Role 3 | Not started |  |
| EDA, splits, demo pairs found | Role 1 | Not started |  |
| 9-model comparison, class-wise analysis (E2, E3) | Role 1 | Not started |  |
| Calibration (E4) | Role 1 | Not started |  |
| OOD detectors + manufactured-unknowns evaluation (E5, E6) | Role 2 | Not started |  |
| Stability + risk-coverage (E7, E8) | Role 2 | Not started |  |
| trust\_engine.py + artefacts | Role 2 | Not started |  |
| FastAPI + tests + Postman collection | Role 3 | Not started |  |
| Frontend pages | Role 4 | Not started |  |
| Deployment (backend, frontend) + live Postman run | Roles 3 and 4 | Not started |  |
| Chosen extra features | per Section 8.5 | Not started |  |
| Report, slides, demo rehearsal | Role 1 (report), Role 4 (slides, demo) | Not started |  |

### 9.4 Team members (fill in)

| Role | Name | Location / time zone | Best hours for calls |
| --- | --- | --- | --- |
| Role 1: ML Lead (team lead) |  |  |  |
| Role 2: Trust Engine Engineer |  |  |  |
| Role 3: Backend and DevOps Engineer |  |  |  |
| Role 4: Frontend and Experience Lead |  |  |  |
