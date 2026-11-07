# Role 4: Frontend and Experience Lead: website, usability, demo

Also read: AGENTS.md. Handbook sections: 2 (Role 4), 6.5, 6.7, 6.8, 7.5, 7.8, 8.3 (A and C).

## Mission
Make the trust behaviour visible and easy to use on a phone, and run the demo. The faculty named JavaScript, so this role carries that requirement.

## You own (edit freely)
frontend/ (React + Vite recommended), the translations file, mock-driven development, frontend hosting (Vercel, Netlify or GitHub Pages), the in-browser familiarity check, the demo script, the slides, the backup video.

## Tasks, in order (pick one at a time)
1. Week 1: scaffold React + Vite (or plain JavaScript if nobody can learn React in a few days); one place for the API address (VITE_API_URL, baked in at build time); deploy a blank page; test it on a phone.
2. Build against frontend/mock/predict_examples.json and the Postman mock server until the real API is ready, then switch the address.
3. Recommend page: seven inputs (sliders plus number boxes with typical-range hints), verdict banner (icon plus word plus colour, never colour alone), top-3 probability bars, trust meter, stability, expandable "Why?", risk-appetite switch (Conservative, Balanced, Bold), "Try typical input" and "Try impossible input" buttons, download report.
4. Fetch with loading, error and "waking the server" states; AbortController plus 300 ms debounce so a stale reply never overwrites a newer one; show field-level messages from 422 replies.
5. What-if page: sliders that re-call the API live, counterfactual hints, climate stress test (call /predict twice, original and shifted temperature and rainfall, then compare; no new route needed).
6. Trust displays: comfort-zone bars, "What to measure next" (shown on Caution), conformal set if present, nearest real examples, nutrient gap view (describe the gap only; no fertiliser quantities).
7. Model Arena page (GET /arena): comparison table and charts, per-crop heatmap, reliability diagram, E5 AUROC table. Model Card page (GET /model-info).
8. Batch page: CSV upload, table of verdicts with flagged rows highlighted, downloadable report, friendly errors for 400 and 413.
9. In-browser familiarity check: load ood_browser.json from Role 2 and compute Mahalanobis in JavaScript as sliders move; the server stays the final authority.
10. Usability: mobile-first, accessible colours, all text in one translations file (English first, then Telugu and Hindi if time allows, checked by a native speaker), feedback button ("Was this wrong?").
11. Demo: script from handbook 7.8 using inputs taken from Role 1's data plots; evidence slides with Role 2's figures; backup video and screenshots; rehearse twice on a phone.
12. Deploy to the host, set the API address as a build-time variable, and give Role 3 the exact origin for CORS.

## You hand over
Mock-driven site early, live site URL, demo script, slides, backup video.

## You need
Frozen contract and mock JSON (day 2), the Postman mock server, ood_browser.json, arena_metrics.json, demo inputs from Role 1, evidence figures from Role 2.

## Role-specific rules for the AI
- Build to the frozen contract only; never invent response fields. If you need a new field, ask.
- Verdict first, numbers second, reasons third. Never rely on colour alone.
- Never hard-code the API address; a missing environment variable should fail loudly, not fall back to localhost in production.
- Always show loading and error states; free hosts can take up to a minute to wake.
- Demo inputs must come from the team's own data and plots, not from the guide.
- Translations need a native speaker's check before they are shown to judges.

## First prompt to paste
> I'm Role 4 (Frontend and Experience Lead). Read AGENTS.md, docs/roles/ROLE4_Frontend_Experience.md, and handbook sections 2 (Role 4), 6.5, 6.7 and 7.5. Don't write code yet. Summarise my tasks, the files I own, what I need from others, and what you'd build first. Then wait for me to pick one task.
