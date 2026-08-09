# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

This is **not** a normal application repo you freely edit. It is a **guided TDD rebuild course**: the backend of `paper-rag-agent-main` (an academic-paper RAG system) is being reconstructed here from scratch, one file at a time, in runnable dependency order. The sibling `../paper-rag-agent-main` repository is the **read-only source baseline** — never write new code there. All new code lives in this directory (`paper-rag-agent-rebuild`).

Equivalence with the baseline is judged by runtime behavior, public interfaces, tests, and eval gates — **not** byte-level file identity.

## Session-resume protocol (read this first, every session)

Before doing anything else:

1. Read `LEARNING_STATE.md` — the current checkpoint, constraints, and the single next step.
2. Read `AGENTS.md` if present, and obey it.
3. For the active feature, read its design + plan under `docs/superpowers/specs/` and `docs/superpowers/plans/`.
4. Run read-only `git status --short` and `git log -5 --oneline`. **The filesystem and git history are the source of truth** — `LEARNING_STATE.md` can lag behind git. If the state file's "next step" contradicts committed history, reconcile against git (verify which plan tasks are already committed) and proceed from the real next uncommitted step. Do **not** reset, checkout, clean, or overwrite any uncommitted file or demo data to make reality match the stale pointer.

Do not re-run requirements interviews or restart from the baseline repo.

## Division of labor (strict)

- **The user (learner)** writes and modifies all production files under `src/paper_rag/`, and runs **all** git commands, installs, tests, demos, and services.
- **The assistant (you)** directly create/modify test files (`tests/test_*.py`) and demo scripts (`scripts/demo_*.py`) — no per-edit permission needed. You must **not** write production `src/` files yourself; instead give the learner the exact per-segment code and explanation to type.
- You may edit and commit `LEARNING_STATE.md` **only** when the user explicitly says to "update progress" (提交进度), and that commit must contain no other files.
- Teach and implement **one file at a time**; a test is the file's up-front acceptance contract.

## Forced acceptance protocol (for anything with external side effects)

Features touching dependency boundaries or external I/O must proceed in this exact order, each step gating the next:

1. **Boundary test** — may use mocks; pins the interface (inputs, outputs, exceptions, dependency calls). Not proof of completion.
2. **Production implementation** — minimal code to pass the boundary test.
3. **Real demo** — a runnable `scripts/demo_*.py` using real services / real data / public APIs, printing the data flow step by step, failing via assertions + non-zero exit. Must use isolated temp data or a dedicated collection and clean up after itself.
4. **Real integration test** — a **mock-free** `tests/test_*_real.py`, run with `uv run pytest -vv -s <file>`. If a service/model/key is unavailable it must **fail loudly** — never `skip` and then claim acceptance.
5. **Checkpoint** — only after boundary test + real demo + real integration test + Ruff all pass may the feature be committed.

Real acceptance uses real resources: real temp SQLite files, real Docker/embedded Qdrant with isolated collections, real MinerU models/CUDA, real APIs. Pure functions may use unit tests alone but must be covered in a real end-to-end demo. Never let an empty result (blank Markdown, page-marker-only text) masquerade as success.

## TDD loop per task

Each plan task: observe **RED** → write minimal implementation → run focused tests + Ruff → commit. Plans in `docs/superpowers/plans/` are checkbox-tracked, file-by-file, and pin exact model revisions — do not float to `main`.
