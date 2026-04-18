# PLAN-v1.1 — DETB v1 Release Gap Closure

> Written 2026-04-18. Snapshot-dated. Supersedes no prior plan (no previous `PLAN*.md` in repo).
> Canonical reference for v1 scope: [`docs/v1-acceptance-checklist.md`](docs/v1-acceptance-checklist.md).

## 1. Context

### What DETB is today

DETB is a reproducible Isaac Lab experiment runner for quadruped design studies. It
emits structured evidence (manifests, episode metrics, aggregate metrics,
candidate requirements, artifact registry) from a fixed five-command **v1 support
contract**: `train`, `evaluate`, `visualize`, `bundle-artifacts`,
`generate-requirements`. Six **experimental** commands remain in the CLI
(`train-gui`, `sweep`, `sensor-eval`, `terrain-eval`, `failure-eval`, `tune`) but
are explicitly out of v1 scope. Pinned baseline: Isaac Sim `5.1.0`, Isaac Lab
`2.3.0` at `../IsaacLab-5.1`, conda env `isaaclab51`, default backend `mock`.

### What's wrong

The repo is mid-flight on a coherent v1-hardening pass with substantial
uncommitted work (12 files, +487/-112):

- New `docs/v1-acceptance-checklist.md` defines the v1 gate.
- `detb/cli.py:23-38` adds `V1_SUPPORTED_COMMANDS` / `EXPERIMENTAL_COMMANDS`
  tuples and a help epilog.
- `tests/test_pipeline.py` adds +204 lines of v1 contract tests (train→evaluate
  →bundle→generate end-to-end, visualize-execute artifact contract).
- `source/detb_lab/detb_lab/runtime.py:46-53` adds a new
  `resolve_policy_module(runner)` helper; play/eval scripts refactored to call
  it.
- `.github/workflows/ci.yml` renames the job to `mock_v1`, adds a 20-minute
  timeout, pip cache, and a `detb train --set execution.backend=mock` smoke
  step.
- `README.md`, `EXECUTION_GUIDE.md`, `Operator_Instructions.md`, `docs/cli.md`
  restructured to split v1 vs experimental consistently.

Three exploration passes (CLI/pipeline, simulator adapter, tests/docs/CI)
identified nine genuine gaps remaining after that pending work lands, plus one
gap that is "land the pending work itself":

| Gap | Summary | Severity | Scope |
|-----|---------|----------|-------|
| A | CI enforces only `pytest` + mock `train`; checklist requires `evaluate`, `visualize`, `bundle-artifacts`, `generate-requirements` too | v1 blocker | In v1 |
| B | `bundle_artifacts()` and `generate_requirements()` have no isolated unit tests — tested only inside the end-to-end contract test | v1 quality | In v1 |
| C | `detb/cli.py:23-38` tuples and `cli.py:54-66` argparse `choices` are duplicated, kept in sync by discipline only | v1 quality | In v1 |
| D | `"summary.md"` literal repeats 21× across `detb/pipeline.py` + `detb/artifacts.py`; artifact type map inlined at `pipeline.py:68-79`; near-duplicate `ArtifactRecord` entries across 7 handlers | Maintenance cost | In v1 |
| E | No test enforces consistency between CLI command tuples and argparse `choices` | v1 quality | In v1 |
| F | `detb/pipeline.py` is 832 lines / 23 top-level functions | Post-v1 refactor | **Deferred** |
| G | `scripts/detb_isaaclab_play.py::main()` is 246 lines with nested rollout/telemetry/fault logic | Post-v1 refactor | **Deferred** |
| H | Experimental commands have no isolated tests (one bulk integration test only) | Explicitly out of v1 | **Deferred** |
| I | `EXECUTION_GUIDE.md` and `Operator_Instructions.md` lack artifact-list detail for `bundle-artifacts` / `generate-requirements` | Doc alignment | In v1 |
| J | The uncommitted v1-hardening pass itself is not committed | Blocks everything | Phase 0 |

### Outcome

After Phase 5, the v1 support contract is fully locked: every v1 command has an
isolated unit test, CI fails fast on any v1-checklist regression, the CLI has
a single source of truth for its command taxonomy with a drift-guard test, and
the operator-facing docs agree on artifact inventories. Phases 6 catalogs the
post-v1 work so it doesn't rot.

### Captured decisions

Three ambiguous decisions surfaced during planning. Auto-mode defaults were
taken; the user can override in review.

**Decision D-1 — Commit topology for the pending v1-hardening work (Phase 0).**
Default: **three logical commits** in this order:
1. Docs (`README.md`, `EXECUTION_GUIDE.md`, `Operator_Instructions.md`,
   `docs/cli.md`, new `docs/v1-acceptance-checklist.md`)
2. CLI tuples (`detb/cli.py`)
3. Runtime helper + script refactor + tests + CI (`source/detb_lab/detb_lab/runtime.py`,
   `scripts/detb_isaaclab_*.py`, `tests/test_pipeline.py`, `tests/test_runtime.py`,
   `.github/workflows/ci.yml`)

Rationale: isolates the mechanical script refactor and the new tests into a
single bisect-friendly commit; docs-only commit and CLI-only commit are
independently revertable without affecting the runtime behavior.

**Decision D-2 — When to raise CI strictness to the full checklist.** Default:
**Phase 2**, after Phase 1 proves each v1 handler is individually verifiable.
Keeps Phase 0 narrowly scoped to "land what's already written."

**Decision D-3 — Experimental-command CLI treatment.** Default: **status quo**
(remain in the CLI with epilog disclosure). The tuples stay as organizational
metadata, not access control. Post-v1 work can revisit `--enable-experimental`
gating if the user wants stricter surface control.

---

## 2. Phased Plan

Ordering rule (user-stated): **cleanup/tokenization before deletes,
decomposition before feature work, CLI simplification before backend swap**.
Each phase lands independently; CI must stay green after each merge.

### Phase 0 — Land the pending v1-hardening work

**Goal.** Commit what's already in the working tree as three reviewable commits
(per D-1). Nothing new; strictly what `git status` already shows.

**Files touched (all already modified).**

| File | Lines Δ (approx) | Intent |
|------|-------------------|--------|
| `.github/workflows/ci.yml` | +13 / −2 | `mock_v1` job, 20-min timeout, pip cache, mock train smoke |
| `detb/cli.py` | +22 / −6 | `V1_SUPPORTED_COMMANDS`, `EXPERIMENTAL_COMMANDS`, epilog |
| `source/detb_lab/detb_lab/runtime.py` | +10 / 0 | `resolve_policy_module()` at lines 46-53 |
| `scripts/detb_isaaclab_common.py` | +2 | Export `resolve_policy_module` |
| `scripts/detb_isaaclab_eval.py` | +7 / −2 | Call helper |
| `scripts/detb_isaaclab_play.py` | +5 / −1 | Call helper |
| `tests/test_pipeline.py` | +204 | v1 contract + visualize-execute tests |
| `tests/test_runtime.py` | +8 / −2 | Policy-module-resolution test |
| `README.md` | +61 / −23 | Split v1 vs experimental sections |
| `EXECUTION_GUIDE.md` | +52 / −21 | v1 baseline operator path added |
| `Operator_Instructions.md` | +49 / −18 | v1 contract + bundle/generate added |
| `docs/cli.md` | +56 / −27 | Supported vs experimental restructure |
| `docs/v1-acceptance-checklist.md` | +32 (new) | Canonical v1 gate |

**New helpers.** None beyond what's already staged.

**Risk / mitigation.** Mixed 12-file diff; review difficulty. Mitigation: the
three-commit split isolates docs, CLI taxonomy, and runtime+tests+CI.

**Verification.**
- `python -m pytest -q` green locally (Windows, `isaaclab51` conda env).
- GitHub Actions `mock_v1` job green on the branch push.
- `python -m detb.cli train --set execution.backend=mock` exits 0 and prints a
  run dir under `outputs/train/`.

**Closes gaps.** J. Enables C and E (tuples now exist to deduplicate against).

---

### Phase 1 — Isolated unit tests for `bundle_artifacts` and `generate_requirements`

**Goal.** One focused test per v1 command handler so a regression in either
function surfaces independently of the +65-line integration test at
`tests/test_pipeline.py:33-97`.

**Files touched.**
- `tests/test_pipeline.py` — appended, ~+80 lines. Do **not** modify the v1
  contract test; add siblings next to it.

**New helpers (test-only).**
- `_seed_eval_run_dir(tmp_path, *, with_terrain=False, with_failure=False) -> Path`
  — writes a minimal valid `run_manifest.json` + `aggregate_metrics.csv`
  (+ optional `terrain_eval.json`, `failure_eval.json`) without invoking
  `run_evaluate`. Round-trips through `RunManifest(**...).to_dict()` so a field
  rename in `detb/models.py` breaks the seed loudly.

**New tests.**
- `test_bundle_artifacts_rebuilds_summary_from_csv(tmp_path)` — seeds manifest +
  aggregate CSV, calls `bundle_artifacts`, asserts `summary.md` contains
  `"Rebuilt from stored aggregate metrics."` and `"## Aggregate Metrics"`
  (matches the strings already asserted at `tests/test_pipeline.py:75-77`).
- `test_bundle_artifacts_missing_manifest_raises(tmp_path)` — clear error path.
- `test_generate_requirements_no_evidence_emits_req_0000(tmp_path)` — verifies
  the DETB-REQ-0000 fallback (see `detb/pipeline.py:315-328`).
- `test_generate_requirements_emits_req_0001_when_threshold_met(tmp_path)` —
  seeds `aggregate_metrics.csv` with a `task_success_rate` row that passes the
  thresholds at `detb/pipeline.py:343-359`.
- `test_generate_requirements_emits_req_0002_from_terrain_eval(tmp_path)` —
  seeds `terrain_eval.json`.
- `test_generate_requirements_emits_req_0003_from_failure_eval(tmp_path)` —
  seeds `failure_eval.json`.

**Risk / mitigation.** Seed helper drifts from the real `run_evaluate` schema.
Mitigation: round-trip through `RunManifest.to_dict()`.

**Verification.** `python -m pytest -q tests/test_pipeline.py -k "bundle_artifacts or generate_requirements"` — six new tests pass; pre-existing contract test
unchanged and green.

**Closes gaps.** B.

---

### Phase 2 — CI enforces the full v1 acceptance checklist

**Goal.** CI exits non-zero if any v1 command fails, not just `pytest` + `train`.
`.github/workflows/ci.yml` currently stops at the mock train smoke (line 25);
this phase extends it.

**Files touched.**
- `.github/workflows/ci.yml` — ~+20 / 0. Keep the `mock_v1` job name; append
  steps after the existing "Mock train smoke".

**New steps (each a distinct GHA step for failure attribution).**

1. `Mock evaluate smoke` — `python -m detb.cli evaluate --set execution.backend=mock --set execution.output_root="${{ runner.temp }}/detb-outputs"`; capture the
   printed run dir into `$GITHUB_ENV` as `DETB_EVAL_RUN_DIR`.
2. `Mock visualize (non-exec) smoke` — `python -m detb.cli visualize --set execution.backend=mock --set visualization.execute=false --set execution.output_root="${{ runner.temp }}/detb-outputs"`; assert a
   `visualize_command.json` exists in the printed run dir.
3. `Bundle-artifacts smoke` — `python -m detb.cli bundle-artifacts --source-dir "$DETB_EVAL_RUN_DIR"`; assert `summary.md` exists.
4. `Generate-requirements smoke` — `python -m detb.cli generate-requirements --source-dir "$DETB_EVAL_RUN_DIR"`; assert `requirement_ledger.json` exists.

All assertions use `test -f` bash-style on `ubuntu-latest` — no new Python
scripts. `runner.temp` is already in use at `ci.yml:25`, so no new path logic
is introduced.

**New helpers.** None.

**Risk / mitigation.**
- Windows-authored path separators bleeding into CI — already mitigated: the
  workflow uses `runner.temp` and POSIX idioms.
- CI minute budget — mock backend is sub-second per command; 20-minute timeout
  stays untouched. Pytest runs first so unit failures gate the rest.

**Verification.** Push branch; `mock_v1` job exercises all four new steps
successfully. Re-run each command locally on Windows to confirm parity.

**Closes gaps.** A.

---

### Phase 3 — Single-source CLI command registry + drift-guard test

**Goal.** Eliminate the duplication between the tuples at `detb/cli.py:23-38`
and the hard-coded `choices=[...]` list at `detb/cli.py:54-66`. Add a test
file (`tests/test_cli.py` does not exist today) to prevent future drift.

**Files touched.**
- `detb/cli.py` — ~+15 / −14. Replace the hard-coded `choices=[...]` with a
  generated list that preserves current help-text order (`train`, `evaluate`,
  `bundle-artifacts`, `sweep`, `sensor-eval`, `terrain-eval`, `failure-eval`,
  `generate-requirements`, `visualize`, `train-gui`, `tune`).
- `tests/test_cli.py` — **new**, ~+50 lines.

**New helpers.**
- In `detb/cli.py`:
  - `_ALL_COMMANDS: tuple[str, ...]` — explicit order tuple preserving the
    current help-text layout. Derived from `V1_SUPPORTED_COMMANDS` and
    `EXPERIMENTAL_COMMANDS` but with the exact ordering currently in
    `choices=[...]` so `--help` output is byte-stable for anyone scripting
    against it.
  - The `argparse` call changes to `choices=list(_ALL_COMMANDS)`.
- In `tests/test_cli.py`:
  - `test_argparse_choices_match_command_tuples()` — imports `_parser`, walks
    the positional `command` action, asserts its `choices` set equals
    `set(V1_SUPPORTED_COMMANDS) | set(EXPERIMENTAL_COMMANDS)`.
  - `test_v1_and_experimental_are_disjoint()` — asserts
    `set(V1_SUPPORTED_COMMANDS).isdisjoint(EXPERIMENTAL_COMMANDS)`.
  - `test_every_command_has_a_dispatch_branch()` — regex-scans `detb/cli.py`
    source for `if command == "<name>"` for each tuple member. Fails if a new
    command is added to a tuple without a dispatch branch. Source-level check
    is cheaper and safer than invoking `main([cmd])` with a real pipeline.

**Risk / mitigation.** Reordering `choices` silently changes `--help` output.
Mitigation: `_ALL_COMMANDS` preserves the exact current order.

**Verification.**
- `python -m pytest -q tests/test_cli.py` — three new tests pass.
- `python -m detb.cli --help` output diff-clean against pre-change baseline.
- `python -m detb.cli train --set execution.backend=mock` still works.

**Closes gaps.** C, E.

---

### Phase 4 — Tokenize artifact-name literals in `pipeline.py` / `artifacts.py`

**Goal.** Replace the 21 `"summary.md"` occurrences, the eight
`ArtifactRecord` description strings that collide across handlers, and the
function-local `mapping` at `detb/pipeline.py:68-79` with named module-level
constants. **No functional change; byte-identical artifact names.**

**Files touched.**
- `detb/artifacts.py` — introduce the constants (this is the lower-level
  module; `pipeline.py` already imports from it). ~+12 lines.
- `detb/pipeline.py` — swap literals for constants; hoist `_artifact_type_for_path`
  mapping to a module constant. ~+5 / −15.

**New helpers (all `Final[...]` constants).**
- `SUMMARY_MD_NAME: Final[str] = "summary.md"`
- `REQUIREMENT_LEDGER_CSV: Final[str] = "requirement_ledger.csv"`
- `REQUIREMENT_LEDGER_JSON: Final[str] = "requirement_ledger.json"`
- `CANDIDATE_REQUIREMENTS_MD: Final[str] = "candidate_requirements.md"`
- `ARTIFACT_TYPE_BY_SUFFIX: Final[dict[str, str]]` — module-level, replacing
  the function-local `mapping` at `pipeline.py:68-79`.
- `_summary_artifact(description: str) -> ArtifactRecord` (in `pipeline.py`) —
  thin constructor returning `ArtifactRecord("markdown", SUMMARY_MD_NAME, description)`. Collapses the eight near-duplicate records at `pipeline.py:155,
  202, 246, 303, 530, 579, 674` etc.

**Risk / mitigation.** A typo in a constant silently changes a filename,
violating the "do not change artifact names" non-goal. Mitigation:
- Phase 1 unit tests and the existing v1 contract test assert exact filenames.
- Run full `python -m pytest -q` after each constant substitution, not only at
  the end.
- The git diff should show **only** literal-to-constant swaps — any control-flow
  or filename change is a review red flag.

**Verification.** `python -m pytest -q` fully green; `git diff` on
`pipeline.py` / `artifacts.py` shows only substitutions.

**Closes gaps.** D.

---

### Phase 5 — Doc alignment (EXECUTION_GUIDE + Operator_Instructions)

**Goal.** `EXECUTION_GUIDE.md` and `Operator_Instructions.md` list the same
artifact inventories for `bundle-artifacts` and `generate-requirements` that
`README.md` does.

**Files touched.**
- `EXECUTION_GUIDE.md` — add detail blocks for the two commands in the
  "Artifact Expectations" section. ~+25 / −5.
- `Operator_Instructions.md` — expand the "Expected outputs" under each of
  the two command subsections. ~+20 / −5.
- `docs/cli.md` — spot-check; already updated in Phase 0.

**New helpers.** None — docs only.

**Risk / mitigation.** Docs drift from reality. Mitigation: copy the artifact-
list language directly from `README.md` after Phase 0 merged; cite the same
filenames asserted by Phase 1 unit tests.

**Verification.** Manual diff review against `README.md` and
`docs/v1-acceptance-checklist.md:7-11`. No automated check.

**Closes gaps.** I.

---

### Phase 6 — Post-v1 deferral catalog

**Goal.** Record gaps F, G, H as tracked post-v1 work so they don't rot.

**Files touched.**
- `docs/v1-acceptance-checklist.md` — add a "Deferred to Post-V1" section
  (~+15 lines) naming:
  - `detb/pipeline.py` modularization (gap F) — current size 832 lines / 23
    functions.
  - `scripts/detb_isaaclab_play.py::main()` decomposition (gap G) — current
    246 lines.
  - Experimental-command isolated test coverage (gap H).

**New helpers.** None.

**Risk / mitigation.** The catalog goes stale. Mitigation: cite the file and
approximate line ranges at the time of writing; a post-v1 audit PR refreshes
them.

**Verification.** Review-only; no tests.

**Closes gaps.** Documents F, G, H as deferred.

---

## 3. Critical Files

| File | Role | Touched By |
|------|------|-----------|
| `detb/cli.py` | CLI entrypoint, command taxonomy | Phase 0, 3 |
| `detb/pipeline.py` | Command orchestration, artifact writing | Phase 4 |
| `detb/artifacts.py` | Summary + plot writers, artifact name strings | Phase 4 |
| `detb/models.py` | `RunManifest`, `ArtifactRecord`, `RequirementRecord` dataclasses | Phase 1 (read only) |
| `tests/test_pipeline.py` | v1 contract + unit tests | Phase 0, 1 |
| `tests/test_cli.py` | **New** — command-tuple drift guard | Phase 3 |
| `tests/test_runtime.py` | Runtime script helper coverage | Phase 0 |
| `.github/workflows/ci.yml` | v1 gate in CI | Phase 0, 2 |
| `docs/v1-acceptance-checklist.md` | Canonical v1 gate definition | Phase 0, 6 |
| `source/detb_lab/detb_lab/runtime.py` | Adapter helpers for Isaac scripts | Phase 0 |
| `scripts/detb_isaaclab_{common,play,eval}.py` | Isaac Lab entry scripts | Phase 0 |
| `README.md`, `EXECUTION_GUIDE.md`, `Operator_Instructions.md`, `docs/cli.md` | Operator-facing docs | Phase 0, 5 |

## 4. Reusable Code Already In Tree

- `detb/io.py::create_manifest`, `write_manifest_bundle`, `read_csv`,
  `read_json` — already the artifact IO primitives; Phase 1 seed helper reuses
  them.
- `detb/models.py` dataclasses — already carry `.to_dict()` serializers; no
  parallel schemas need to be invented.
- `detb/artifacts.py::rebuild_summary` — already the implementation behind
  `bundle_artifacts`; Phase 1 tests call `bundle_artifacts` (public surface)
  rather than reaching into it.
- `detb/config.py::load_config`, `default_config_dir` — already drive the
  Hydra composition; Phase 1 seed tests reuse them via the existing `_cfg()`
  helper at `tests/test_pipeline.py:22-27`.
- `V1_SUPPORTED_COMMANDS`, `EXPERIMENTAL_COMMANDS` tuples — Phase 3 builds on
  these rather than introducing a third category system.

## 5. End-to-End Verification After All Phases

After Phase 5 lands, this sequence must succeed from a clean tree on Windows
with `conda activate isaaclab51`:

```powershell
python -m pip install -e .[dev]
python -m pip install -e source/detb_lab
python -m pytest -q                                       # all tests, incl. Phase 1 and Phase 3 additions
python -m detb.cli train --set execution.backend=mock     # v1 train
python -m detb.cli evaluate --set execution.backend=mock  # v1 evaluate; capture run dir
python -m detb.cli visualize --set visualization.execute=false    # v1 visualize preview
python -m detb.cli bundle-artifacts --source-dir outputs/evaluate/<run_id>
python -m detb.cli generate-requirements --source-dir outputs/evaluate/<run_id>
```

And in CI (Ubuntu):

- `mock_v1` job green on push/PR to `main`.
- Job log shows all Phase 2 steps reached and exit-0.

## 6. Rough Sizing

| Phase | Scope | Size |
|-------|-------|------|
| 0 | Land pending uncommitted v1-hardening work (three commits) | S |
| 1 | Isolated unit tests for `bundle_artifacts` + `generate_requirements` | M |
| 2 | CI enforces full v1 checklist | S |
| 3 | CLI single-source + drift-guard test | S |
| 4 | Tokenize literals in `pipeline.py` / `artifacts.py` | M |
| 5 | Doc alignment (EXECUTION_GUIDE / Operator_Instructions) | S |
| 6 | Post-v1 deferral catalog | S |

S = under 2 hours. M = 2–5 hours. L = 5+ hours (none).
Total: ~1 focused day, seven independently-mergeable PRs.

## 7. Non-Goals (Explicit)

- **Do not** refactor `detb/pipeline.py` (832 lines, 23 top-level functions)
  into multiple modules. Post-v1 (gap F).
- **Do not** decompose `scripts/detb_isaaclab_play.py::main()` (246 lines).
  Post-v1 (gap G).
- **Do not** add isolated unit tests for `train-gui`, `sweep`, `sensor-eval`,
  `terrain-eval`, `failure-eval`, or `tune`. Out of v1 scope per
  [`docs/v1-acceptance-checklist.md:20-27`](docs/v1-acceptance-checklist.md).
- **Do not** change the pinned baseline (Isaac Sim `5.1.0`, Isaac Lab `2.3.0`,
  conda env `isaaclab51`). See [`AGENTS.md:15-21`](AGENTS.md).
- **Do not** change artifact names or data contracts. Constants introduced in
  Phase 4 are verbatim copies of current string literals; unit tests catch
  accidental drift.
- **Do not** introduce simulator imports (`isaaclab`, `omni`, `isaacsim`,
  `pxr`, `rsl_rl`) into `detb/pipeline.py`, `detb/artifacts.py`, `detb/stats.py`,
  or any test module. Isolation rule at [`AGENTS.md:44-47`](AGENTS.md).
- **Do not** add Windows-only path logic. CI runs `ubuntu-latest`; dev is
  Windows. Stay on `pathlib` + POSIX idioms.
- **Do not** introduce a plugin/entry-point mechanism for commands. The tuples
  in `detb/cli.py` are the registry.

## 8. Handoff Notes

- This plan assumes the user's current Git user/repo state: branch `main`,
  12 files modified and not staged, untracked new file
  `docs/v1-acceptance-checklist.md`. Phase 0 is a no-op if those already
  landed; the remaining phases remain valid.
- CI runs on `ubuntu-latest` with Python 3.11; dev runs on Windows 11 with the
  `isaaclab51` conda env. Keep the two paths parity-safe.
- The Isaac Lab baseline (`5.1.0` / `2.3.0`) is pinned in `AGENTS.md`. Any
  simulator-version change is out-of-scope for this plan.
