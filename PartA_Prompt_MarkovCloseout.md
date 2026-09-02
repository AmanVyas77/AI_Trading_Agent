# PART A — Close out the Markov exit layer (git only)

You are working in the AI Trading Agent repo. This task is **pure repository
hygiene**. Do not modify any source file, do not run any backtest, do not
retrain anything.

## HARD RULE — read before anything else

**You must not execute any git command that writes.** That means `git add`,
`git commit`, `git push`, `git cherry-pick`, `git checkout`, `git switch`,
`git reset`, `git restore`, `git stash`, `git rm`, `git mv`, `git tag`, and
`git filter-branch` are all forbidden, however convenient they look.

You may run exactly these read-only commands, and nothing else that starts with
`git`: `git rev-parse`, `git branch --show-current`, `git branch --contains`,
`git log`, `git status`, `git diff`, `git show`.

Everything that writes gets **printed as a copy-paste block for me to run
myself**. If you find yourself reasoning that staging is harmless, or that it
would be easier to just commit and let me amend it — stop, and print the command
instead. If you think a write command is genuinely necessary and not covered
above, ask me first and wait.

## Repo guard — run this FIRST and stop if it fails

```bash
cd ~/dev/"Ai Trading Agent"
git rev-parse --show-toplevel
git branch --show-current
git log --oneline -1
```

**STOP and report** unless the toplevel ends in `Ai Trading Agent`, the branch is
`prototype/markov-exit-layer`, and HEAD is `510916c`. This project has a history
of a nested decoy repo and of hash rewrites — if any of the three differ, do not
proceed, just tell me what you found.

## Context

The Markov exit layer (Zhang + Andrade) was under diagnosis from 2026-07-26 to
2026-08-19. Prompt 2 ran on 2026-08-19 and the pre-committed decision rule fired:

- Condition (a) required a NEGATIVE mean `ret_full` on SELL-flagged positions.
  Measured: **+0.1617**, hit rate 9/29. Condition (a) FAILS.
- Andrade-driven sells: 2 correct out of 17, one-sided p = **0.0012** —
  significantly anti-predictive, not merely unhelpful.
- Zhang-gated sells: 12, mean +0.0041, p = 0.387 — break-even, and structurally
  inert (45/45 Case I have `p0 >= x*` at min margin 33.7×; Case II is 0/225 and
  unreachable since `f1` min 2.592 vs `rho` 0.03).
- Conditions (b) and (c) were deliberately NOT tested — (a) failing is sufficient
  for SHELVE, and testing them would be searching for grounds to reopen a settled
  question.

Verdict: **SHELVE**. The layer is not in production and never was. The verdict
report is written but untracked, and that is the only thing left to do.

## Task 1 — inventory the working tree

```bash
git status --short
```

Expected modified: `.gitignore`, `MarkovExit_Diagnostics_Prompts_2026-07-30.md`,
`Sprint7_Prompts.md`, `backtests/exit_layer_reconciliation_2026-07-30.md`,
`memory/future_ideas.md`, `research_wiki/_index.md`,
`scripts/backtest_exit_layer.py`, `src/exit/exit_manager.py`.

Report anything present that is NOT on that list before going further.

## Task 2 — review the two source diffs

```bash
git diff scripts/backtest_exit_layer.py src/exit/exit_manager.py
```

These are diagnostic instrumentation from Prompts 1C/1D/2 (the `--vintage` flag
plumbing, the `_assert_interpreter()` guard, and the verdict block in
`write_report`). Confirm for me in a few lines:

1. Nothing here changes the LIVE path. `src/live/scorer.py`, `rebalance.py`,
   `paper_runner.py` and `broker_alpaca.py` must be untouched.
2. Nothing here changes entry-model behaviour or the frozen pickle.
3. `get_live_regime_signal` is still called in exactly two places —
   `src/live/rebalance.py:140` and `src/exit/exit_manager.py:249` — and neither
   is inside a backtest loop. Verify with
   `grep -rn get_live_regime_signal src/`.

If any of those three is false, STOP and tell me. Otherwise continue.

## Task 3 — size check before staging

The frozen vintage directory is 639 MB and must never enter git history.

```bash
du -sh backtests/vintage_2026-08-04/
grep -n "vintage" .gitignore
```

If `backtests/vintage_2026-08-04/` is not covered by `.gitignore`, tell me — I
would rather add the ignore rule than rely on remembering not to stage it.

## Task 4 — print the commands, do not run them

House rule in this repo: **you never run `git add`, `git commit`, or
`git push`.** You print exact copy-paste commands and I run them myself. No
`Co-Authored-By` trailer.

Print a single fenced block I can paste, which:

1. Stages the two source files plus `scripts/freeze_vintage.py`.
2. Stages the 2026-08-19 verdict report and its `_gen.py`, the 2026-08-19
   decision analysis and its `_gen.py`, the 2026-08-04 units-and-attribution
   report with its `_gen.py` and `_console.txt`, the 2026-08-04
   vintage-and-andrade-off report, and the 2026-07-30 reconciliation.
3. Stages the modified prompt/notes files: the diagnostics prompt batch, the
   2026-08-11 news-backfill fix prompt, `Sprint7_Prompts.md`,
   `memory/future_ideas.md`, `research_wiki/`, `.gitignore`.
4. Explicitly does NOT stage `backtests/vintage_2026-08-04/`, `data/`, `logs/`,
   or `models/`.
5. Includes two verification lines to run *before* the commit — a count of
   staged files and `git diff --cached --stat | tail -1` — so I can confirm no
   large blob slipped in.
6. Then the `git commit -m` with a message recording the verdict and the numbers
   above, and `git push origin prototype/markov-exit-layer`.

## Task 5 — the main-branch question

`510916c` ("Sprint 9: fix AV news-backfill silent failures") is a genuine
pipeline fix that landed on this prototype branch rather than `main`. Check
whether it is on `main`:

```bash
git log --oneline main..prototype/markov-exit-layer
git branch --contains 510916c
```

Tell me whether a `git cherry-pick 510916c` onto `main` is still needed, and if
so print that command too — separately, so I can decide independently. Do not
switch branches.

## Out of scope

Do not run Prompts 3 or 4 of the diagnostic batch. Do not tune Andrade's
confidence threshold or the regime detector on the 2025-01 → 2026-06 window —
17 months of in-sample search is not evidence, and the batch instructions
forbid it. Do not delete `backtests/vintage_2026-08-04/`; it is the reproducible
input for every number in the reports.
