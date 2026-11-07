# Role 3: Backend and DevOps Engineer: API, tests, deployment

Also read: AGENTS.md. Handbook sections: 2 (Role 3), 3, 6 (all), 7.5, 8.2 (location autofill).

## Mission
Turn the engine into a reliable, tested, deployed REST API that the website calls. Everything must work on a public URL, not just a laptop.

## You own (edit freely)
backend/main.py (routes only), requirements.txt, Dockerfile, .dockerignore, backend/tests/test_api.py, .github/workflows/tests.yml, postman/, docs/api-contract.md, README, the repo skeleton, backend hosting, request logging, the drift monitor endpoint.

## Tasks, in order (pick one at a time)
1. Day 1: folder layout (handbook 6.10), .gitignore (venv, .env), branch protection (pull request plus one review), README skeleton.
2. Day 1 to 2: write docs/api-contract.md and the three example replies (recommend, caution, refuse) with Roles 4 and 2, then freeze. Save to frontend/mock/predict_examples.json. Decide with the team whether weather-only mode is a query parameter (for example mode=weather_only) before the freeze.
3. Week 1: deploy a hello endpoint (/health) in Docker to the chosen host, fix CORS for the chosen frontend address, create a Postman collection with a {{base_url}} variable, and a Postman mock server from the three example replies for Role 4.
4. Routes in main.py: GET /health, GET /model-info, POST /predict?risk=, POST /predict/batch, POST /explain, GET /arena, GET /monitor. Additive extras: POST /feedback and a weather helper for location autofill.
5. Pydantic CropInput with physical bounds (not training min/max); field-level 422 handler; 413 for files over 1 MB; 400 for a missing column or more than 500 rows; never return raw Python errors.
6. Load the engine once at start-up; use plain def for prediction routes; include model_version in replies.
7. Logging: one JSON line per request (time, verdict, OOD percentile, stability, latency, model version, the seven inputs; no personal data); rate limiting (slowapi); CORS allow-list of your own origins only.
8. Tests: pytest for health, impossible pH (422), valid prediction, plus every case in the handbook 6.6 table; GitHub Actions on every push.
9. Postman: all cases with test scripts, Collection Runner, export collection and environment JSON, screenshots for the report.
10. Dockerfile reading ${PORT:-8000}; run locally; deploy (Render, Hugging Face Spaces, Railway or Fly.io; check current free-tier rules); verify /docs on the public URL.
11. Weather helper: wrap Open-Meteo or NASA POWER after Role 1 confirms the rainfall period and units; read each service's terms.
12. Drift monitor: GET /monitor with flag rate and PSI per feature using Role 2's psi() and Role 1's reference_distribution.json. Say plainly that free hosts may reset the disk.
13. E10: median /predict time locally and on the live server. Final live Postman run, README with API reference and deploy steps.

## You hand over
Repo skeleton and contract (day 2), Postman mock server and live hello endpoint (week 1), real API (end of phase 2), live URL with CORS correct for Role 4.

## You need
TrustEngine class from Role 2 (stub first), model files from Role 1, arena_metrics.json for /arena, Role 4's final frontend address for CORS.

## Role-specific rules for the AI
- A Refuse verdict is HTTP 200; pH = 40 is HTTP 422. Keep the two apart.
- Never commit secrets; use environment variables. Never leave allow_origins as ["*"] in the final version.
- Cap upload size and rows; never return raw Python errors to users.
- Pin versions; use the same requirements.txt locally, in CI and in Docker.
- Do not edit trust_engine.py or the model files; ask their owners.
- The contract is frozen: additive routes are fine, changes to existing fields need the whole team.

## First prompt to paste
> I'm Role 3 (Backend and DevOps Engineer). Read AGENTS.md, docs/roles/ROLE3_Backend_DevOps.md, and handbook sections 2 (Role 3), 3, 6 and 7.5. Don't write code yet. Summarise my tasks, the files I own, what I need from others, and what you'd build first. Then wait for me to pick one task.
