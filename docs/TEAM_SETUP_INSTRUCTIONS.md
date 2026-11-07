Markdown · TEAM_SETUP_INSTRUCTIONS.md

# Crop Trust Engine: Team Setup and Workflow Instructions

For the four team members and for the AI assistant (for example Antigravity) each member uses. Repo: [https://github.com/Asritha2008/crop-trust-engine](https://github.com/Asritha2008/crop-trust-engine)

## What this file is

The step-by-step guide to setting up the repo, working with an AI assistant, and proceeding as a team. It works together with these files, which must be in the repo:

- `AGENTS.md` (repo root): rules every AI assistant must follow
- `docs/PROJECT_GUIDE.md`: the full project handbook (concepts, stack, design, features, code)
- `docs/roles/ROLE1_ML_Lead.md`, `ROLE2_Trust_Engine.md`, `ROLE3_Backend_DevOps.md`, `ROLE4_Frontend_Experience.md`: one briefing per role
- `docs/decisions.md` and `docs/PROGRESS.md`: the team's shared memory (templates at the end of this file)

## The four roles

| Role | Lane | Main deliverables |
| --- | --- | --- |
| Role 1: ML Lead (team lead) | Data and models | EDA, 9-model comparison, class-wise analysis, calibration, weather-only model, reality-check experiment, report editing |
| Role 2: Trust Engine Engineer | Trust layer | OOD detectors and their evaluation, stability, verdict logic, risk profiles, explanations, conformal sets |
| Role 3: Backend and DevOps Engineer | API, tests, deployment | FastAPI routes, validation, Postman collection, pytest, Docker, live backend, logging |
| Role 4: Frontend and Experience Lead | Website and demo | React + Vite pages, what-if sliders, in-browser familiarity check, Arena and Model Card pages, demo and slides |

Write each person's name here once decided: Role 1 \_\_\_\_ , Role 2 \_\_\_\_ , Role 3 \_\_\_\_ , Role 4 \_\_\_\_ .

---

## Part 1: Set up the repo (once, about 30 minutes; one person does it, ideally Role 3)

1. Put `AGENTS.md` in the repo root.
2. Create `docs/roles/` and put the four role files in it.
3. Put the project handbook in as `docs/PROJECT_GUIDE.md`.
4. Create `docs/decisions.md` and `docs/PROGRESS.md` from the templates at the end of this file.
5. Commit and push to `main`.
6. Everyone else clones the repo and opens the folder in the AI tool.

## Part 2: How to make the AI understand everything

AI assistants do not remember earlier chats, and long documents get skimmed. So:

1. **Layer the context.** `AGENTS.md` holds the rules for everyone, your role file holds your tasks, and the handbook is reference material read only where the role file points.
2. **Start every new chat the same way.** Paste your role's first prompt (below). A fresh chat has forgotten everything, so do this every time.
3. **Check its understanding before it writes code.** Compare its summary with your role file and correct it. Then ask: "What are the three biggest mistakes to avoid in my role?" A good answer mentions data leakage, invented numbers, and editing other people's files.
4. **Give it one task at a time.** Paste one numbered task from your role file. Never say "do my whole role".
5. **Make it cite its sources.** Ask which handbook section it is following. If it cannot say, it is guessing.
6. **Keep the shared memory up to date.** After each finished task add two lines to `docs/PROGRESS.md`. Record every decision (for example a chosen threshold) in `docs/decisions.md`. Tell the assistant to read both at the start of every session.
7. **Run the code yourself.** "It works" from the AI is not evidence. Run it, read the output, then commit.

### First prompt to paste (replace N and the role name)

> I'm Role N (role name). Read AGENTS.md, docs/roles/ROLEN\_.md, docs/decisions.md and docs/PROGRESS.md, and the handbook sections my role file points to. Don't write code yet. Summarise my tasks, the files I own, what I need from other roles, and what you'd build first. Then wait for me to pick one task.

### Prompt for each task

> Task: . Follow AGENTS.md. Say which handbook section you are following. Keep the code plain and explainable. Run it, show the output, add a test or a saved result, then stop. Do not touch files owned by other roles.

### Instructions for the AI assistant reading this file

- Read `AGENTS.md` and the person's role file before anything else.
- Do not write code until the person has confirmed your summary and picked a task.
- Never invent numbers or results; only report what code you ran produced.
- Never change a locked decision or another role's files without asking.
- If something in the handbook looks wrong or untested, say so instead of guessing.

---

## Part 3: Order of work

1. **Kickoff call (1 hour, everyone).** Agenda:
   1. Name each person against Roles 1 to 4.
   2. Freeze the API contract and the three example replies (recommend, caution, refuse).
   3. Decide whether weather-only mode is a query parameter on /predict.
   4. Dataset path: A (Kaggle only), B (Kaggle plus outside reality checks, recommended) or C (merge a richer dataset).
   5. Pick the 3 to 5 extra features (recommended list: handbook Section 8.5).
   6. React + Vite or plain JavaScript; backend and frontend hosts.
   7. First draft of the noise model and the three risk profiles (placeholders until tuned).
   8. Meeting rhythm and where daily updates are posted. Record everything in `docs/decisions.md`.
2. **Week 1: setup, all four in parallel.**
   - Role 1: data checks, train/validation/test splits committed (day 3), baseline pipeline committed (day 4).
   - Role 2: class interface agreed with Role 3, stub `trust_engine.py` returning the three mock replies (day 3).
   - Role 3: repo skeleton, contract document, mock JSON, hello endpoint deployed, CORS fixed, Postman collection and mock server.
   - Role 4: scaffold the site, deploy a blank page, wire the Recommend page to the mock JSON.
   - Gate: contract frozen, hello endpoint live, everyone has run everything locally.
3. **Phase 1: build your lane** (about 55% of the effort). Work through your role file's tasks in order. One branch and one pull request per task. Review partners: Roles 1 and 2 review each other; Roles 3 and 4 review each other.
4. **Phase 2: integrate** (about 15%). Role 3 plugs in the real engine and deploys; Role 4 switches to the live API; Role 2 delivers `ood_browser.json` for the in-browser check. Gate: an end-to-end run from a phone on the public URL, and the Postman collection passes live.
5. **Phase 3: extras and polish** (about 10%). Each extra needs one measurable claim and one slide of evidence. If a lane falls behind, cut in this order: conformal sets, batch CSV, drift monitor, Telugu and Hindi. Never cut the must-haves: model comparison, class-wise report, calibration, stability, working OOD flag with three verdicts, API, website, deployment.
6. **Phase 4: report, rehearsal, buffer** (about 10%). Each person writes their own chapter; Role 1 merges. Rehearse the demo twice on a phone and record a backup video.

## Part 4: Working from four places

- **Folder ownership.** Edit only the files your role owns (listed in `AGENTS.md`). Need a change elsewhere? Ask the owner or open a pull request.
- **Git.** Branch per task named `role<N>/<topic>` (for example `role2/ood-detectors`). Small commits with clear messages. Merge through pull requests with one reviewer. `main` must always run. Never commit secrets or large data files.
- **Notebooks.** One owner per notebook; strip outputs before committing (nbstripout); save figures to `docs/figures/`. Anything the API imports is a plain `.py` file.
- **Daily update** in the team chat, three lines: done, doing next, blocked.
- **Two 30-minute calls a week:** Monday plan, Thursday demo of what works. Share screens and run the code.
- **Weekly teach-back:** one member explains their layer for ten minutes, so everyone can answer the mentor's questions.

## Part 5: Handoffs between roles

| Handoff | From | To | When |
| --- | --- | --- | --- |
| Contract and mock JSON | Role 3 (with 2 and 4) | Roles 2, 4 | day 2 |
| Split files and seed | Role 1 | Role 2 | day 3 |
| Stub engine | Role 2 | Role 3 | day 3 |
| Baseline pipeline | Role 1 | Roles 2, 3 | day 4 |
| Postman mock and live hello endpoint | Role 3 | Role 4 | week 1 |
| Final pipeline | Role 1 | Roles 2, 3 | end of phase 1 |
| Real engine and artefacts | Role 2 | Role 3 | end of phase 1 |
| ood_browser.json | Role 2 | Role 4 | phase 2 |
| arena_metrics.json | Roles 1, 2 | Roles 3, 4 | rolling |
| Live API URL | Role 3 | Role 4 | end of phase 2 |
| Evidence figures | Roles 1, 2 | Role 4 | phase 3 |

## Part 6: Pitfalls to avoid

- Two people editing the same file causes merge conflicts. Respect folder ownership.
- Changing the JSON contract casually breaks the other lanes. Tell all four people first.
- Numbers in the handbook are placeholders. Only numbers produced by your own code on your own data go in the report.
- Data leakage: split first; fit everything on training or validation data only; touch the test set once at the end.
- A refusal is HTTP 200. Invalid input such as pH 40 is HTTP 422. Keep the two apart.
- Free hosts sleep when idle: open the live site just before the demo and keep a backup video.
- In the viva every member must be able to explain every layer, not only their own.

---

## Template: docs/decisions.md

```markdown
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
| | Noise model (final values) | | Roles 1, 2 |
| | Risk profile thresholds (final values) | | Role 2 |
```

## Template: docs/PROGRESS.md

```markdown
# Progress log
Add two lines after each finished task: what was done, which files changed.

| Date | Role | Task | Files changed | Notes / open issues |
| --- | --- | --- | --- | --- |
| | | | | |

## Component tracker
| Component | Owner | Status | Notes |
| --- | --- | --- | --- |
| API contract frozen + mock JSON | Role 3 | Not started | |
| EDA, splits, demo pairs found | Role 1 | Not started | |
| 9-model comparison, class-wise analysis | Role 1 | Not started | |
| Calibration | Role 1 | Not started | |
| OOD detectors + manufactured-unknowns evaluation | Role 2 | Not started | |
| Stability + risk-coverage | Role 2 | Not started | |
| trust_engine.py + artefacts | Role 2 | Not started | |
| FastAPI + tests + Postman collection | Role 3 | Not started | |
| Frontend pages | Role 4 | Not started | |
| Deployment (backend, frontend) + live Postman run | Roles 3, 4 | Not started | |
| Chosen extra features | per handbook 8.5 | Not started | |
| Report, slides, demo rehearsal | Role 1 (report), Role 4 (slides, demo) | Not started | |
```